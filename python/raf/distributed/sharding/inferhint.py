# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, unused-argument, missing-function-docstring
"""Implementaion of Infer Hints"""
from queue import PriorityQueue
from typing import Callable, List

from raf._ffi.sharding._make import ShardOpCallAttrs
from raf._ffi.op import GetOp
from raf._lib import _register_func, relay
from raf.distributed.sharding.shardspec import BaseShardSpec, UnsetShardSpec
from raf.distributed.sharding.utils import make_replicated_spec

from .expandrule import (
    ShardInfo,
    always_apply,
    expand_opcall,
    is_same_spec,
    is_sharded,
)
from .expandrule import register_expansion_rule as register_infer_hint


def try_when(cond: Callable[[ShardInfo], bool], priority=1):
    """Specify the priority and the condition when this infer hint should be used.

    Parameters
    ----------
    cond : function(ShardInfo) -> bool
        A function validating this infer hint is eligible to apply.
    """

    if not hasattr(try_when, "counter"):
        try_when.counter = 0
    if not hasattr(try_when, "rules"):
        try_when.rules = {}

    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_names"):
            raise ValueError("Must register infer hint first")
        for op_name in pyfunc.op_names:
            op = GetOp(op_name)
            if op not in try_when.rules:
                try_when.rules[op] = PriorityQueue()
            try_when.rules[op].put((-priority, try_when.counter, cond, pyfunc))
            try_when.counter += 1
        return pyfunc

    return decorator


@_register_func("raf.sharding._infer_shardspec")
def infer_shardspec(call: relay.Call):
    # pylint: disable=too-many-locals, too-many-branches
    """Fill the placeholders of ShardSpec with infer hints."""
    rules = try_when.rules[call.op]
    s = ShardInfo(call)

    # Step 1: Inherit ShardSpec from previous output
    inherit_sin = []
    for i in range(len(s.sin)):
        if isinstance(s.sin[i], UnsetShardSpec):
            if isinstance(s.args[i], relay.Call) and hasattr(s.args[i].attrs, "sin"):
                # cannot use isinstance to check the type of OpCall Attrs
                # direct inherit ShardSpec
                prev_sinfo = ShardInfo(s.args[i])
                inherit_sin.append(prev_sinfo.sout[0])
            else:
                # the previous output doesn't have ShardSpec
                ndim = len(s.args[0].checked_type.concrete_shape)
                inherit_sin.append(make_replicated_spec(ndim))

        else:
            # already exist a specified ShardSpec
            inherit_sin.append(s.sin[i])

    inherit_attrs = ShardOpCallAttrs(inherit_sin, s.sout)
    inherit_call = relay.Call(s.op, s.args, inherit_attrs)
    inherit_s = ShardInfo(inherit_call)

    # Step 2: Match InferHints
    filled_calls = []
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(inherit_s):
            filled_calls.extend([relay.Call(s.op, s.args, a) for a in irgen(inherit_s)])
    if not filled_calls:
        raise ValueError("Failed to match an InferHint")

    # Step 3: Check the solution is practicable
    ninputs = len(s.sin)
    noutputs = len(s.sout)
    immut_in_idx = [i for i in range(ninputs) if is_sharded(s.sin[i]) and not s.sin[i].mutable]
    immut_out_idx = [i for i in range(noutputs) if is_sharded(s.sout[i]) and not s.sout[i].mutable]

    possible_calls = []
    for filled_call in filled_calls:
        if not expand_opcall(filled_call):
            # there doesn't exist a expansion rule that accepts this sharding solution
            continue
        filled_s = ShardInfo(filled_call)
        immut_args = [(inherit_s.sin[i], filled_s.sin[i]) for i in immut_in_idx] + [
            (inherit_s.sout[i], filled_s.sout[i]) for i in immut_out_idx
        ]
        for pair in immut_args:
            if not is_same_spec(pair[0], pair[1]):
                # violate immutable flag
                break
        else:
            possible_calls.append(filled_call)

    # Step 4: Pick an OpCall with full ShardSpec
    # TODO: should use graph searching algorithm with cost map here.
    # For now, always select the first solution.
    inferred_call = possible_calls[0]
    inferred_s = ShardInfo(inferred_call)

    # Step 5: Insert Reshard OpCalls
    resharded_args = []
    for i in range(ninputs):
        if is_same_spec(inherit_s.sin[i], inferred_s.sin[i]):
            resharded_args.append(inferred_s.args[i])
        else:
            resharded_args.append(
                relay.Call(
                    GetOp("raf.op._reshard"),
                    [inferred_s.args[i]],
                    ShardOpCallAttrs([inherit_s.sin[i]], [inferred_s.sin[i]]),
                )
            )

    print("[Sharding Infer] %s %s ### %s" % (inherit_s.op, inferred_s.attrs, inherit_s.attrs))
    return relay.Call(inferred_s.op, resharded_args, inferred_s.attrs)


def is_unset(s: BaseShardSpec):
    """Check whether it is an UnsetShardSpec (placeholder of ShardSpec)."""
    return isinstance(s, UnsetShardSpec)


@try_when(always_apply)
@register_infer_hint(["raf.op.add", "raf.op.subtract"])
def element_wise_op_with_2in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sin[1], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [ShardOpCallAttrs([e, e], [e]) for e in specs]


@try_when(always_apply)
@register_infer_hint(["raf.op.relu"])
def element_wise_op_with_1in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [ShardOpCallAttrs([e], [e]) for e in specs]
