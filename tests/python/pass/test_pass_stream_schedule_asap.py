# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
from typing import Dict, List
import pytest
import tvm
import tvm.relay
import mnm
from mnm.testing import randn
from mnm.ir.pass_manager import MNMSequential
from mnm._ffi.pass_ import ToGraphNormalForm, ASAPStreamSchedule
from mnm._core.ir_ext import extended_var
from mnm.ir import ScopeBuilder


class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators: Dict[str, tvm.ir.Op] = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = tvm.relay.op.get(f"mnm.op.{op_name}")
        return self.operators[op_name]

    def make_tuple(self, fields: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let("", tvm.relay.Tuple(fields))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let("", tvm.relay.Call(self.get_operator(op_name), args))

    def set_stream(self, device_id: int, stream_id: int) -> tvm.relay.Var:
        return self.call("set_stream", [mnm.ir.const(device_id), mnm.ir.const(stream_id)])

    def add_event(self, event_id: int) -> tvm.relay.Var:
        return self.call("add_event", [mnm.ir.const(event_id)])

    def wait_event(self, event_id: int) -> tvm.relay.Var:
        return self.call("wait_event", [mnm.ir.const(event_id)])

    def atan(self, x: tvm.ir.RelayExpr) -> tvm.relay.Var:
        return self.call("atan", [x])

    def concatenate(self, x: tvm.ir.RelayExpr, axis: int) -> tvm.relay.Var:
        return self.call("concatenate", [x, mnm.ir.const(axis)])

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_asap_schedule_simple_branches():
    class Model(mnm.Model):
        """
         ┌───────x──────┐
         │       │      │
         ▼       ▼      ▼
        atan0  atan1  atan2
         │       │      │
         │       ▼      ▼
         │     atan3  atan4
         │       │      │
         │       │      ▼
         └───┐   │    atan5
             │   │   ┌───
             ▼   ▼   ▼
            concatenate
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)  # atan 0

            p_1 = mnm.atan(x)  # atan 1
            p_1 = mnm.atan(p_1)  # atan 3

            p_2 = mnm.atan(x)  # atan 2
            p_2 = mnm.atan(p_2)  # atan 4
            p_2 = mnm.atan(p_2)  # atan 6
            return mnm.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = MNMSequential([ToGraphNormalForm(), ASAPStreamSchedule()])(mod)

    def expected():
        """
        fn (%x: Tensor[(2, 2), float32]) {
          let %x_0 = mnm.op.set_stream(int64(0), int64(1));
          let %x_1 = mnm.op.atan(%x);
          let %x_2 = mnm.op.atan(%x_1);
          let %x_3 = mnm.op.atan(%x_2);
          let %x_4 = mnm.op.add_event(int64(0));
          let %x_5 = mnm.op.set_stream(int64(0), int64(2));
          let %x_6 = mnm.op.atan(%x);
          let %x_7 = mnm.op.atan(%x_6);
          let %x_8 = mnm.op.add_event(int64(1));
          let %x_9 = mnm.op.set_stream(int64(0), int64(3));
          let %x_10 = mnm.op.atan(%x);
          let %x_11 = mnm.op.wait_event(int64(1));
          let %x_12 = mnm.op.wait_event(int64(0));
          let %x_13 = (%x_10, %x_7, %x_3);
          let %x_14 = mnm.op.concatenate(%x_13, int64(0));
          %x_14
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)  # atan 2
        x_2 = sb.atan(x_1)  # atan 4
        x_3 = sb.atan(x_2)  # atan 6
        x_4 = sb.add_event(0)
        x_5 = sb.set_stream(0, 2)
        x_6 = sb.atan(x)  # atan 1
        x_7 = sb.atan(x_6)  # atan 3
        x_8 = sb.add_event(1)
        x_9 = sb.set_stream(0, 3)
        x_10 = sb.atan(x)  # atan 0
        x_11 = sb.wait_event(1)
        x_12 = sb.wait_event(0)
        x_13 = sb.make_tuple([x_10, x_7, x_3])
        x_14 = sb.concatenate(x_13, 0)
        return tvm.relay.Function([x], sb.ret(x_14))

    # To draw the dataflow graph with schedule, uncomment the following line:
    # mnm.utils.visualizer.draw_dataflow_graph(mod['main'], 'sb.png', draw_event_nodes=True)
    assert tvm.ir.structural_equal(mod["main"], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_asap_schedule_branch_in_branch():
    class Model(mnm.Model):
        """
         ┌───────x──────┐
         │       │      │
         ▼       ▼      ▼
        atan0   atan1  atan2
         │       │      │
         │       ▼      └─┐
         │   ┌──atan3─┐   │
         │   │        │   ▼
         │   ▼        ▼  atan6
         │  atan4    atan5│
         │   │        │   │
         │   ▼        ▼   │
         │   concatenate0 │
         │       │        ▼
         └───┐   │   ┌───atan7
             │   │   │
             ▼   ▼   ▼
            concatenate1
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)  # atan 0

            p_1 = mnm.atan(x)  # atan 1
            p_1 = mnm.atan(p_1)  # atan 3
            p_1a = mnm.atan(p_1)  # atan 4
            p_1b = mnm.atan(p_1)  # atan 6
            p_1 = mnm.concatenate([p_1a, p_1b])

            p_2 = mnm.atan(x)  # atan 2
            p_2 = mnm.atan(p_2)  # atan 6
            p_2 = mnm.atan(p_2)  # atan 7
            return mnm.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = MNMSequential([ToGraphNormalForm(), ASAPStreamSchedule()])(mod)

    def expected():
        """
        fn (%x: Tensor[(2, 2), float32]) {
          let %x_0 = mnm.op.set_stream(int64(0), int64(1));
          let %x_1 = mnm.op.atan(%x);
          let %x_2 = mnm.op.atan(%x_1);
          let %x_3 = mnm.op.add_event(int64(0));
          let %x_4 = mnm.op.atan(%x_2);
          let %x_5 = mnm.op.set_stream(int64(0), int64(2));
          let %x_6 = mnm.op.wait_event(int64(0));
          let %x_7 = mnm.op.atan(%x_2);
          let %x_8 = mnm.op.add_event(int64(1));
          let %x_9 = mnm.op.set_stream(int64(0), int64(3));
          let %x_10 = mnm.op.atan(%x);
          let %x_11 = mnm.op.atan(%x_10);
          let %x_12 = mnm.op.atan(%x_11);
          let %x_13 = mnm.op.add_event(int64(2));
          let %x_14 = mnm.op.set_stream(int64(0), int64(1));
          let %x_15 = mnm.op.wait_event(int64(1));
          let %x_16 = (%x_4, %x_7);
          let %x_17 = mnm.op.concatenate(%x_16, int64(0));
          let %x_18 = mnm.op.add_event(int64(3));
          let %x_19 = mnm.op.set_stream(int64(0), int64(4));
          let %x_20 = mnm.op.atan(%x);
          let %x_21 = mnm.op.wait_event(int64(3));
          let %x_22 = mnm.op.wait_event(int64(2));
          let %x_23 = (%x_20, %x_17, %x_12);
          let %x_24 = mnm.op.concatenate(%x_23, int64(0));
          %x_24
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)  # atan 1
        x_2 = sb.atan(x_1)  # atan 3
        x_3 = sb.add_event(0)
        x_4 = sb.atan(x_2)  # atan 5
        x_5 = sb.set_stream(0, 2)
        x_6 = sb.wait_event(0)
        x_7 = sb.atan(x_2)  # atan 4
        x_8 = sb.add_event(1)
        x_9 = sb.set_stream(0, 3)
        x_10 = sb.atan(x)  # atan 2
        x_11 = sb.atan(x_10)  # atan 6
        x_12 = sb.atan(x_11)  # atan 7
        x_13 = sb.add_event(2)
        x_14 = sb.set_stream(0, 1)
        x_15 = sb.wait_event(1)
        x_16 = sb.make_tuple([x_4, x_7])
        x_17 = sb.concatenate(x_16, 0)  # concat 0
        x_18 = sb.add_event(3)
        x_19 = sb.set_stream(0, 4)
        x_20 = sb.atan(x)  # atan 0
        x_21 = sb.wait_event(3)
        x_22 = sb.wait_event(2)
        x_23 = sb.make_tuple([x_20, x_17, x_12])
        x_24 = sb.concatenate(x_23, 0)  # concat 1
        return tvm.relay.Function([x], sb.ret(x_24))

    # mnm.utils.visualizer.draw_dataflow_graph(mod['main'], 'bnb.png', draw_event_nodes=True)
    assert tvm.ir.structural_equal(mod["main"], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_asap_schedule_stacked_blocks():
    class Model(mnm.Model):
        """
         ┌─────x────┐
         │     │    │
         │     │    ▼
         │     │   atan2
         ▼     ▼    │
        atan0 atan1 ▼
         │     │   atan3
         │     │    │
         ▼     ▼    ▼
         concatenate0
         │     │    │
         │     │    ▼
         │     │   atan6
         ▼     ▼    │
        atan4 atan5 ▼
         │     │   atan7
         │     │    │
         ▼     ▼    ▼
         concatenate1
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)  # atan 0
            p_1 = mnm.atan(x)  # atan 1
            p_2 = mnm.atan(x)  # atan 2
            p_2 = mnm.atan(p_2)  # atan 3
            x = mnm.concatenate([p_0, p_1, p_2])  # concat 0
            p_0 = mnm.atan(x)  # atan 4
            p_1 = mnm.atan(x)  # atan 5
            p_2 = mnm.atan(x)  # atan 6
            p_2 = mnm.atan(p_2)  # atan 7
            return mnm.concatenate([p_0, p_1, p_2])  # concat 1

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = MNMSequential([ToGraphNormalForm(), ASAPStreamSchedule()])(mod)

    def expected():
        """
        fn (%x: Tensor[(2, 2), float32]) {
          let %x_0 = mnm.op.set_stream(int64(0), int64(1));
          let %x_1 = mnm.op.atan(%x);
          let %x_2 = mnm.op.atan(%x_1);
          let %x_3 = mnm.op.add_event(int64(0));
          let %x_4 = mnm.op.set_stream(int64(0), int64(2));
          let %x_5 = mnm.op.atan(%x);
          let %x_6 = mnm.op.add_event(int64(1));
          let %x_7 = mnm.op.set_stream(int64(0), int64(3));
          let %x_8 = mnm.op.atan(%x);
          let %x_9 = mnm.op.wait_event(int64(1));
          let %x_10 = mnm.op.wait_event(int64(0));
          let %x_11 = (%x_8, %x_5, %x_2);
          let %x_12 = mnm.op.concatenate(%x_11, int64(0));
          let %x_13 = mnm.op.add_event(int64(2));
          let %x_14 = mnm.op.atan(%x_12);
          let %x_15 = mnm.op.atan(%x_14);
          let %x_16 = mnm.op.add_event(int64(3));
          let %x_17 = mnm.op.set_stream(int64(0), int64(2));
          let %x_18 = mnm.op.wait_event(int64(2));
          let %x_19 = mnm.op.atan(%x_12);
          let %x_20 = mnm.op.add_event(int64(4));
          let %x_21 = mnm.op.set_stream(int64(0), int64(1));
          let %x_22 = mnm.op.wait_event(int64(2));
          let %x_23 = mnm.op.atan(%x_12);
          let %x_24 = mnm.op.wait_event(int64(4));
          let %x_25 = mnm.op.wait_event(int64(3));
          let %x_26 = (%x_23, %x_19, %x_15);
          let %x_27 = mnm.op.concatenate(%x_26, int64(0));
          %x_27
        }
        """
        sb = ANFBuilder()

        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)  # atan 2
        x_2 = sb.atan(x_1)  # atan 3
        x_3 = sb.add_event(0)
        x_4 = sb.set_stream(0, 2)
        x_5 = sb.atan(x)  # atan 1
        x_6 = sb.add_event(1)
        x_7 = sb.set_stream(0, 3)
        x_8 = sb.atan(x)  # atan 0
        x_9 = sb.wait_event(1)
        x_10 = sb.wait_event(0)
        x_11 = sb.make_tuple([x_8, x_5, x_2])
        x_12 = sb.concatenate(x_11, 0)  # concat 0
        x_13 = sb.add_event(2)
        x_14 = sb.atan(x_12)  # atan 6
        x_15 = sb.atan(x_14)  # atan 7
        x_16 = sb.add_event(3)
        x_17 = sb.set_stream(0, 2)
        x_18 = sb.wait_event(2)
        x_19 = sb.atan(x_12)  # atan 5
        x_20 = sb.add_event(4)
        x_21 = sb.set_stream(0, 1)
        x_22 = sb.wait_event(2)
        x_23 = sb.atan(x_12)  # atan 4
        x_24 = sb.wait_event(4)
        x_25 = sb.wait_event(3)
        x_26 = sb.make_tuple([x_23, x_19, x_15])
        x_27 = sb.concatenate(x_26, 0)  # concat 1
        return tvm.relay.Function([x], sb.ret(x_27))

    # mnm.utils.visualizer.draw_dataflow_graph(mod['main'], 'stack.png', draw_event_nodes=True)
    assert tvm.ir.structural_equal(mod["main"], expected())


if __name__ == "__main__":
    pytest.main([__file__])
