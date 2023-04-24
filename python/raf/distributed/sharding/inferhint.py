# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, unused-argument, missing-function-docstring
"""Implementaion of Infer Hints"""
from queue import PriorityQueue
from typing import Callable, List

from raf._ffi.sharding._make import ShardOpCallAttrs
from raf._ffi.op import GetOp
from raf._lib import _register_func, relay
from raf.distributed.sharding.shardspec import BaseShardSpec, UnsetShardSpec, ShardSpec
from raf.distributed.sharding.utils import make_replicated_spec

from .expandrule import (
    ShardInfo,
    always_apply,
    expand_opcall,
    is_exact_same_spec,
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
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Fill the placeholders of ShardSpec with infer hints."""
    rules = try_when.rules[call.op]
    s = ShardInfo(call)

    # Step 1: Inherit input spec from previous output

    # inherit_sin should be the correct specs of current inputs
    inherit_sin = []
    # specified_sin should be the user-specified specs with filled unset shard specs
    specified_sin = []

    for i in range(len(s.sin)):
        if isinstance(s.args[i], relay.Call) and hasattr(s.args[i].attrs, "sin"):
            # cannot use isinstance to check the type of OpCall Attrs
            # direct inherit ShardSpec
            prev_sinfo = ShardInfo(s.args[i])
            inherit_sin.append(prev_sinfo.sout[0])
        else:
            # the previous output isn't annotated with ShardSpec
            if isinstance(s.sin[i], ShardSpec):
                # already exist a specified ShardSpec
                inherit_sin.append(s.sin[i])
            else:
                # assume the previous output is replicated on all ranks
                ndim = len(s.args[i].checked_type.concrete_shape)
                inherit_sin.append(make_replicated_spec(ndim))

        if isinstance(s.sin[i], UnsetShardSpec):
            specified_sin.append(inherit_sin[-1])
        else:
            specified_sin.append(s.sin[i])

    inherit_s = s.make_updated(sin=inherit_sin)
    specified_s = s.make_updated(sin=specified_sin)

    # Step 2: Match InferHints

    filled_s_list: List[ShardInfo] = []  # TODO: try to remove duplicated solutions
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(specified_s):
            filled_s_list.extend([s.make_updated(attrs=a) for a in irgen(specified_s)])
        if cond(inherit_s):
            filled_s_list.extend([s.make_updated(attrs=a) for a in irgen(inherit_s)])

    if not filled_s_list:
        raise ValueError("Failed to match an InferHint")

    # Step 3: Check the solution is practicable
    ninputs = len(s.sin)
    noutputs = len(s.sout)
    immut_in_idx = [i for i in range(ninputs) if is_sharded(s.sin[i]) and not s.sin[i].mutable]
    immut_out_idx = [i for i in range(noutputs) if is_sharded(s.sout[i]) and not s.sout[i].mutable]

    possible_s_list: List[ShardInfo] = []
    for filled_s in filled_s_list:
        if not expand_opcall(filled_s.call):
            # there doesn't exist a expansion rule that accepts this sharding solution
            continue
        immut_args = [(inherit_s.sin[i], filled_s.sin[i]) for i in immut_in_idx] + [
            (inherit_s.sout[i], filled_s.sout[i]) for i in immut_out_idx
        ]
        for pair in immut_args:
            if not is_same_spec(pair[0], pair[1]):
                # violate immutable attribute of shard spec
                break
        else:
            # reset Mutable flag for outputs to prevent from spreading this flag mistakenly
            sout = [
                spec if spec.mutable else spec.make_updated(mutable=True) for spec in filled_s.sout
            ]
            possible_s_list.append(filled_s.make_updated(sout=sout))

    # Step 4: Pick an OpCall with full ShardSpec
    # TODO: should use graph searching algorithm with cost map here.
    # For now, always select the first solution.
    inferred_s = possible_s_list[0]

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

    print("[Sharding Infer] OpCall: %s" % s.op)
    for phase in ("In", "Out"):
        for i in range(ninputs if phase == "In" else noutputs):
            if phase == "In":
                a_spec, b_spec, c_spec = s.sin[i], inherit_s.sin[i], inferred_s.sin[i]
            else:
                a_spec, b_spec, c_spec = s.sout[i], inherit_s.sout[i], inferred_s.sout[i]
            print("    %sArg %s: %s" % (phase, i, a_spec), end="")
            if not is_exact_same_spec(a_spec, b_spec):
                print(" -> %s" % b_spec, end="")
            if not is_exact_same_spec(b_spec, c_spec):
                print(" -> %s" % c_spec, end="")
            print()

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
