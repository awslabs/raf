# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, unused-argument
"""Implementation of Expansion Rules"""
import functools
from queue import PriorityQueue
from typing import Callable, List
import numpy as np
import raf
import tvm


from raf._ffi.op import GetOp
from raf._lib import _register_func, relay
from raf.distributed.sharding.shardspec import BaseShardSpec, ShardSpec
from tvm.relay import Expr
from tvm.ir import Op
from tvm.runtime.object import Object

pattern_map = {
    0: "kElemWise",
    1: "kBroadcast",
    2: "kInjective",
    3: "kCommReduce",
    4: "kOutEWiseFusable",
    7: "kTuple",
    8: "kOpaque",
}
# TODO: this pattern map is replicated multiple times in source code


class ShardInfo:
    """Helper for parsing ShardSpec."""

    # pylint: disable=too-few-public-methods
    call: relay.Call
    op: Op
    args: List[Expr]
    attrs: Object
    sin: List[BaseShardSpec]
    sout: List[BaseShardSpec]

    def __init__(self, call: relay.Call):
        assert isinstance(call, relay.Call)
        self.call = call
        self.op = call.op
        self.args = call.args
        self.attrs = call.attrs
        self.sin = call.attrs.sin
        self.sout = call.attrs.sout


def all_satisfied(conds: List[Callable[[ShardInfo], bool]]):
    """Return true when all conditions are satisfied."""

    def func(s: ShardInfo):
        for c in conds:
            if not c(s):
                return False
        return True

    return func


def is_same_spec(*args):
    """Check whether two ShardSpecs are same."""
    for e in args[1:]:
        if not tvm.ir.structural_equal(args[0], e):
            return False
    return True


def is_sharded(s: BaseShardSpec):
    """Check whether it is a ShardSpec."""
    return isinstance(s, ShardSpec)


def is_replicated(s: BaseShardSpec):
    """Check whether it is a replicated ShardSpec."""
    if not isinstance(s, ShardSpec):
        return False
    return s.nshard == 1


def no_subgroup(s: BaseShardSpec):
    """Check whether subgrouping feature is disabled."""
    if not isinstance(s, ShardSpec):
        return False
    return s.ngroup == 1


def always_apply(s: ShardInfo):
    """Always return True."""
    return True


def expand_when(cond: Callable[[ShardInfo], bool], priority=1):
    """Specify the priority and the condition when this expansion rule should be used.

    Parameters
    ----------
    cond : function(ShardInfo) -> bool
        A function validating this expansion rule is eligible to apply.
    """

    if not hasattr(expand_when, "counter"):
        expand_when.counter = 0
    if not hasattr(expand_when, "rules"):
        expand_when.rules = {}

    def decorator(pyfunc):
        if not hasattr(pyfunc, "op_names"):
            raise ValueError("Must register expansion rule first")
        for op_name in pyfunc.op_names:
            op = GetOp(op_name)
            if op not in expand_when.rules:
                expand_when.rules[op] = PriorityQueue()
            expand_when.rules[op].put((-priority, expand_when.counter, cond, pyfunc))
            expand_when.counter += 1
        return pyfunc

    return decorator


def register_expansion_rule(op_name):
    """Register an expansion rule that converts a full-sized op into a partitioned-size op

    Parameters
    ----------
    op_name: str or List[str]
        Names of op to register
    """
    op_names = [op_name] if isinstance(op_name, str) else op_name
    assert isinstance(op_names, list)

    def decorator(pyfunc):
        @functools.wraps(pyfunc)
        def new_pyfunc(call: relay.Call):
            return pyfunc(call)

        setattr(new_pyfunc, "op_names", op_names)
        return new_pyfunc

    return decorator


@_register_func("raf.sharding._match_expansion_rule")
def expand_opcall(call: relay.Call):
    """Match an eligible expansion rule and return expanded IR expr."""
    rules = expand_when.rules[call.op]
    s = ShardInfo(call)
    for rule in rules.queue:
        _, _, cond, irgen = rule
        if cond(s):
            return irgen(s)
    return None


@expand_when(
    all_satisfied([lambda s: is_replicated(s.sin[0]), lambda s: is_sharded(s.sout[0])]),
    priority=1,
)
@register_expansion_rule("raf.op._reshard")
def reshard_replicated_to_sharded(s: ShardInfo):
    """_reshard (R to S) -> strided_slice"""
    begin, end = [], []
    shape = s.args[0].checked_type.concrete_shape
    spec = s.sout[0]
    for idx, dim_nshard, dim_size in zip(spec.logic_index, spec.logic_shape, shape):
        assert dim_size % dim_nshard == 0
        begin.append(int((dim_size // dim_nshard) * idx))
        end.append(int((dim_size // dim_nshard) * (idx + 1)))
    return relay.Call(
        GetOp("raf.op.strided_slice"),
        [
            s.args[0],
            raf.ir.const(begin),
            raf.ir.const(end),
            raf.ir.const([1] * spec.ndim),
            raf.ir.const("end"),
        ],
    )


@expand_when(
    all_satisfied(
        [
            lambda s: print(s.sin[0], s.sout[0]) or True,
            lambda s: is_sharded(s.sin[0]),
            lambda s: is_replicated(s.sout[0]),
        ]
    ),
    priority=1,
)
@register_expansion_rule("raf.op._reshard")
def reshard_sharded_to_replicated(s: ShardInfo):
    """_reshard (S to R) -> allgather"""
    spec = s.sin[0]
    axis = []
    full_shape = []
    for i in range(spec.ndim):
        if spec.logic_shape[i] > 1:
            axis.append(i)
        full_shape.append(int(spec.logic_shape[i]))
        full_shape.append(int(spec.subgroup_shape[i]))
    assert len(axis) == 1  # TODO: remove this constrain
    ranks = np.array([int(e) for e in spec.ranks]).reshape(full_shape)
    nshard_on_dim = int(spec.logic_shape[axis[0]])
    rank_list = np.moveaxis(ranks, axis[0], -1).reshape(
        (ranks.size // nshard_on_dim, nshard_on_dim)
    )
    return relay.Call(
        GetOp("raf.op._allgather"),
        [s.args[0], raf.ir.const(axis[0]), raf.ir.const(rank_list.tolist())],
    )


@expand_when(lambda s: is_same_spec(s.sin[0], s.sin[1], s.sout[0]))
@register_expansion_rule(["raf.op.add", "raf.op.subtract"])
def add_or_sub(s: ShardInfo):
    """add/sub -> add/sub"""
    return relay.Call(s.op, s.args)


@expand_when(lambda s: is_same_spec(s.sin[0], s.sout[0]))
@register_expansion_rule(["raf.op.relu"])  # TODO: should use a generated list instead
def element_wise(s: ShardInfo):
    """element wise -> element wise"""
    return relay.Call(s.op, s.args)


@expand_when(
    all_satisfied(
        [
            lambda s: is_sharded(s.sin[0]) and is_sharded(s.sin[1]),
            lambda s: no_subgroup(s.sin[0]) and no_subgroup(s.sin[1]),
            lambda s: is_replicated(s.sout[0]),
            lambda s: s.sin[0].logic_shape[1] == s.sin[1].logic_shape[0],
        ]
    )
)
@register_expansion_rule(["raf.op.matmul"])
def matmul_algor1(s: ShardInfo):
    """matmul -> matmul + allreduce"""
    y_1 = relay.Call(s.op, s.args)
    y_2 = tvm.relay.Tuple([y_1])
    return relay.Call(GetOp("raf.op._allreduce"), [y_2, raf.ir.const("sum"), raf.ir.const(None)])
