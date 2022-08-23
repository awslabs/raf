# pylint: disable=invalid-name, unused-argument
"""Implementaion of Infer Hints"""
from ctypes import Union
import functools
import numpy as np
import raf
import tvm
from queue import PriorityQueue
from typing import Callable, List, Tuple

from raf._ffi.sharding._make import ShardOpCallAttrs
from raf._ffi.op import GetOp
from raf._lib import _register_func, relay
from raf.distributed.sharding.shardspec import BaseShardSpec, ShardSpec, UnsetShardSpec
from raf.distributed.sharding.utils import make_replicated_spec
from raf._core.value import Value
from raf import distributed as dist
from raf.ir.anf_builder import ANFBuilder
from tvm.relay import Call, Expr
from tvm.ir import Op

from .expandrule import ShardInfo, all_satisfied, always_apply, expand_opcall, is_same_spec, is_sharded
from .expandrule import register_expansion_rule as register_infer_hint

def try_when(cond: Callable[[ShardInfo], bool], priority=1):
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
    rules = try_when.rules[call.op]
    s = ShardInfo(call)

    # Step 1: Inherit ShardSpec from previous output
    filled_sin = []
    for i in range(len(s.sin)):
        if isinstance(s.sin[i], UnsetShardSpec):
            if isinstance(s.args[i], relay.Call) and hasattr(s.args[i].attrs, "sin"):
                # cannot use isinstance to check the type of OpCall Attrs
                # direct inherit ShardSpec
                prev_sinfo = ShardInfo(s.args[i])
                filled_sin.append(prev_sinfo.sout[0])
            else:
                # the previous output doesn't have ShardSpec
                ndim = len(s.args[0].checked_type.concrete_shape)
                filled_sin.append(make_replicated_spec(ndim))

        else:
            # already exist a specified ShardSpec
            filled_sin.append(s.sin[i])
    
    filled_attrs = ShardOpCallAttrs(filled_sin, s.sout)
    filled_call = relay.Call(s.op, s.args, filled_attrs)
    filled_s = ShardInfo(filled_call)

    # Step 2: Match an InferHint
    guessed_calls = []
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(filled_s):
            guessed_calls.extend([relay.Call(s.op, s.args, a) for a in irgen(filled_s)])
    if not guessed_calls:
        raise ValueError("Failed to match an InferHint")

    # Step 3: Check the solution is practicable
    ninputs = len(filled_s.sin)
    noutputs = len(filled_s.sout)
    immut_in_idx = [i for i in range(ninputs) if is_sharded(filled_s.sin[i]) and filled_s.sin[i].mutable == False]
    immut_out_idx = [i for i in range(noutputs) if is_sharded(filled_s.sout[i]) and filled_s.sout[i].mutable == False]

    possible_calls = []
    for guessed_call in guessed_calls:
        if not expand_opcall(guessed_call):
            continue
        guessed_s = ShardInfo(guessed_call)
        immut_args = [(filled_s.sin[i], guessed_s.sin[i]) for i in immut_in_idx] + \
            [(filled_s.sout[i], guessed_s.sout[i]) for i in immut_out_idx]
        for pair in immut_args:
            if not is_same_spec(pair[0], pair[1]):
                break
        else:
            possible_calls.append(guessed_call)

    # Step 4: Pick an OpCall with full ShardSpec
    # TODO: should use graph searching algorithm with cost map here. For now, always select the first solution.
    inferred_call = possible_calls[0]
    inferred_s = ShardInfo(inferred_call)

    # Step 5: Insert Reshard OpCall
    resharded_args = []
    for i in range(ninputs):
        if is_same_spec(filled_s.sin[i], inferred_s.sin[i]):
            resharded_args.append(inferred_s.args[i])
        else:
            resharded_args.append(relay.Call(
                GetOp("raf.op._reshard"),
                [inferred_s.args[i]],
                ShardOpCallAttrs([filled_s.sin[i]], [inferred_s.sin[i]])))
    
    print("[Sharding Infer] %s %s ### %s" % (filled_s.op, inferred_s.attrs, filled_s.attrs))
    return relay.Call(inferred_s.op, resharded_args, inferred_s.attrs)

def is_unset(s: BaseShardSpec):
    return isinstance(s, UnsetShardSpec)

@try_when(always_apply)
@register_infer_hint(["raf.op.add", "raf.op.subtract"])
def element_wise_op_with_2in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sin[1], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [
        ShardOpCallAttrs([e, e], [e]) for e in specs
    ]

@try_when(always_apply)
@register_infer_hint(["raf.op.relu"])
def element_wise_op_with_1in_1out(s: ShardInfo) -> List[ShardOpCallAttrs]:
    specs = []
    for e in (s.sin[0], s.sout[0]):
        if not is_unset(e):
            specs.append(e)
    return [
        ShardOpCallAttrs([e], [e]) for e in specs
    ]
