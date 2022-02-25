# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-import, unused-variable, too-many-locals
from typing import Optional
from typing import Dict, List
import pytest
import tvm
import tvm.relay
import raf
from raf.testing.schedule_verifier import verify_schedule, ExecutionOrderError
from raf._core.ir_ext import extended_var
from raf.ir import ScopeBuilder
from raf.utils.visualizer import draw_dataflow_graph


class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators: Dict[str, tvm.ir.Op] = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = tvm.relay.op.get(f"raf.op.{op_name}")
        return self.operators[op_name]

    def make_tuple(self, fields: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let("", tvm.relay.Tuple(fields))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let("", tvm.relay.Call(self.get_operator(op_name), args))

    def set_stream(self, device_id: int, stream_id: int) -> tvm.relay.Var:
        return self.call("set_stream", [raf.ir.const(device_id), raf.ir.const(stream_id)])

    def add_event(self, event_id: int) -> tvm.relay.Var:
        return self.call("add_event", [raf.ir.const(event_id)])

    def wait_event(self, event_id: int) -> tvm.relay.Var:
        return self.call("wait_event", [raf.ir.const(event_id)])

    def stream_barrier(self) -> tvm.relay.Var:
        return self.call("stream_barrier", [])

    def atan(self, x: tvm.ir.RelayExpr) -> tvm.relay.Var:
        return self.call("atan", [x])

    def concatenate(self, x: tvm.ir.RelayExpr, axis: int) -> tvm.relay.Var:
        return self.call("concatenate", [x, raf.ir.const(axis)])

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()


@pytest.mark.parametrize("removed_events", [[], [0], [1], [0, 1]])
def test_simple_branches_event(removed_events):
    def scheduled_func():
        sb = ANFBuilder()
        x = extended_var("x", shape=[2, 2])
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)
        x_2 = sb.atan(x_1)
        x_3 = sb.atan(x_2)
        if 0 not in removed_events:
            x_4 = sb.add_event(0)
        x_5 = sb.set_stream(0, 1)
        x_6 = sb.atan(x)
        x_7 = sb.atan(x_6)
        if 1 not in removed_events:
            x_8 = sb.add_event(1)
        x_9 = sb.set_stream(0, 2)
        x_10 = sb.atan(x)
        if 1 not in removed_events:
            x_11 = sb.wait_event(1)
        if 0 not in removed_events:
            x_12 = sb.wait_event(0)
        x_13 = sb.make_tuple([x_10, x_7, x_3])
        x_14 = sb.concatenate(x_13, 0)
        return tvm.relay.Function([x], sb.ret(x_14))

    func = scheduled_func()
    # Please uncomment the following line to draw the scheduled dataflow graph
    # draw_dataflow_graph(func, f"./graphs/simple_branches_remove_events_{removed_events}.png",
    #                     draw_event_nodes=True)
    if len(removed_events) > 0:
        with pytest.raises(ExecutionOrderError):
            verify_schedule(func)
    else:
        verify_schedule(func)


@pytest.mark.parametrize("removed_events", [[], [0], [1], [2], [3], [4], [0, 1, 2, 3, 4]])
def test_stacked_blocks_event(removed_events):
    def scheduled_func():
        sb = ANFBuilder()
        x = extended_var("x", shape=[2, 2])
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)  # atan 2
        x_2 = sb.atan(x_1)  # atan 3
        if 0 not in removed_events:
            x_3 = sb.add_event(0)
        x_4 = sb.set_stream(0, 1)
        x_5 = sb.atan(x)  # atan 1
        if 1 not in removed_events:
            x_6 = sb.add_event(1)
        x_7 = sb.set_stream(0, 2)
        x_8 = sb.atan(x)  # atan 0
        if 1 not in removed_events:
            x_9 = sb.wait_event(1)
        if 0 not in removed_events:
            x_10 = sb.wait_event(0)
        x_11 = sb.make_tuple([x_8, x_5, x_2])
        x_12 = sb.concatenate(x_11, 0)  # concat 0
        if 2 not in removed_events:
            x_13 = sb.add_event(2)
        x_14 = sb.atan(x_12)  # atan 6
        x_15 = sb.atan(x_14)  # atan 7
        if 3 not in removed_events:
            x_16 = sb.add_event(3)
        x_17 = sb.set_stream(0, 1)
        if 2 not in removed_events:
            x_18 = sb.wait_event(2)
        x_19 = sb.atan(x_12)  # atan 5
        if 4 not in removed_events:
            x_20 = sb.add_event(4)
        x_21 = sb.set_stream(0, 0)
        if 2 not in removed_events:
            x_22 = sb.wait_event(2)
        x_23 = sb.atan(x_12)  # atan 4
        if 4 not in removed_events:
            x_24 = sb.wait_event(4)
        if 3 not in removed_events:
            x_25 = sb.wait_event(3)
        x_26 = sb.make_tuple([x_23, x_19, x_15])
        x_27 = sb.concatenate(x_26, 0)  # concat 1
        return tvm.relay.Function([x], sb.ret(x_27))

    func = scheduled_func()
    # Please uncomment the following line to draw the scheduled dataflow graph
    # draw_dataflow_graph(func, f"./graphs/stacked_blocks_remove_events_{removed_events}.png",
    #                     draw_event_nodes=True)
    if len(removed_events) > 0:
        with pytest.raises(ExecutionOrderError):
            verify_schedule(func)
    else:
        verify_schedule(func)


@pytest.mark.parametrize("removed_barriers", [[], [0]])
def test_simple_branches_barrier(removed_barriers):
    def scheduled_func():
        sb = ANFBuilder()
        x = extended_var("x", shape=[2, 2])
        x_0 = sb.set_stream(0, 2)
        x_1 = sb.atan(x)
        x_2 = sb.atan(x_1)
        x_3 = sb.atan(x_2)
        x_4 = sb.set_stream(0, 1)
        x_5 = sb.atan(x)
        x_6 = sb.atan(x_5)
        x_7 = sb.set_stream(0, 0)
        x_8 = sb.atan(x)
        if 0 not in removed_barriers:
            x_9 = sb.stream_barrier()
        x_10 = sb.make_tuple([x_8, x_6, x_3])
        x_11 = sb.concatenate(x_10, 0)
        return tvm.relay.Function([x], sb.ret(x_11))

    func = scheduled_func()
    if len(removed_barriers) > 0:
        with pytest.raises(ExecutionOrderError):
            verify_schedule(func)
    else:
        verify_schedule(func)


@pytest.mark.parametrize("removed_barriers", [[], [0], [1], [2], [0, 1, 2]])
def test_stacked_blocks_barrier(removed_barriers):
    def scheduled_func():
        sb = ANFBuilder()
        x = extended_var("x", shape=[2, 2])
        x_0 = sb.set_stream(0, 2)
        x_1 = sb.atan(x)
        x_2 = sb.atan(x_1)
        x_3 = sb.set_stream(0, 1)
        x_4 = sb.atan(x)
        x_5 = sb.set_stream(0, 0)
        x_6 = sb.atan(x)
        if 0 not in removed_barriers:
            x_7 = sb.stream_barrier()
        x_8 = sb.make_tuple([x_6, x_4, x_2])
        x_9 = sb.concatenate(x_8, 0)
        if 1 not in removed_barriers:
            x_10 = sb.stream_barrier()
        x_11 = sb.set_stream(0, 2)
        x_12 = sb.atan(x_9)
        x_13 = sb.atan(x_12)
        x_14 = sb.set_stream(0, 1)
        x_15 = sb.atan(x_9)
        x_16 = sb.set_stream(0, 0)
        x_17 = sb.atan(x_9)
        if 2 not in removed_barriers:
            x_18 = sb.stream_barrier()
        x_19 = sb.make_tuple([x_17, x_15, x_13])
        x_20 = sb.concatenate(x_19, 0)
        return tvm.relay.Function([x], sb.ret(x_20))

    func = scheduled_func()
    if len(removed_barriers) > 0:
        with pytest.raises(ExecutionOrderError):
            verify_schedule(func)
    else:
        verify_schedule(func)


@pytest.mark.parametrize("removed_barriers", [[], [0]])
def test_chain_to_another_chain_barrier(removed_barriers):
    def scheduled_func():
        sb = ANFBuilder()
        x = extended_var("x", shape=[2, 2])
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)
        x_2 = sb.atan(x_1)
        if 0 not in removed_barriers:
            x_3 = sb.stream_barrier()
        x_4 = sb.set_stream(0, 1)
        x_5 = sb.atan(x_2)
        return tvm.relay.Function([x], sb.ret(x_5))

    func = scheduled_func()
    if len(removed_barriers) > 0:
        with pytest.raises(ExecutionOrderError):
            verify_schedule(func)
    else:
        verify_schedule(func)


if __name__ == "__main__":
    pytest.main([__file__])
