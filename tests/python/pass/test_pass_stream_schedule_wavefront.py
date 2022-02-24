# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
from typing import Dict, List
import pytest
import tvm
import tvm.relay
import raf
from raf.testing import randn
from raf.ir.pass_manager import RAFSequential
from raf._ffi.pass_ import ToGraphNormalForm, WavefrontStreamSchedule
from raf._core.ir_ext import extended_var
from raf.ir import ScopeBuilder


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


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_schedule_three_simple_branches():
    class Model(raf.Model):
        # wavefront schedule:
        # wave 1
        #   chain 1: op 1
        #   chain 2: op 2, op 3
        #   chain 3: op 4, op 5, op 6
        # wave 2
        #   chain 1: op 7
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)  # op 1

            p_1 = raf.atan(x)  # op 2
            p_1 = raf.atan(p_1)  # op 3

            p_2 = raf.atan(x)  # op 4
            p_2 = raf.atan(p_2)  # op 5
            p_2 = raf.atan(p_2)  # op 6
            return raf.concatenate([p_0, p_1, p_2])  # op 7

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(opt_level=2, config={"raf.stream_schedule.policy": "wavefront"}):
        mod = RAFSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
        def @main(%x: Tensor[(2, 2), float32]) {
          let %x_0 = raf.op.set_stream(int64(0), int64(0));
          let %x_1 = raf.op.atan(%x);
          let %x_2 = raf.op.set_stream(int64(0), int64(1));
          let %x_3 = raf.op.atan(%x);
          let %x_4 = raf.op.atan(%x_3);
          let %x_5 = raf.op.set_stream(int64(0), int64(2));
          let %x_6 = raf.op.atan(%x);
          let %x_7 = raf.op.atan(%x_6);
          let %x_8 = raf.op.atan(%x_7);
          let %x_9 = raf.op.stream_barrier();
          let %x_10 = raf.op.set_stream(int64(0), int64(0));
          let %x_11 = (%x_1, %x_4, %x_8);
          let %x_12 = raf.op.concatenate(%x_11, int64(0));
          %x_12
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)
        x_2 = sb.set_stream(0, 1)
        x_3 = sb.atan(x)
        x_4 = sb.atan(x_3)
        x_5 = sb.set_stream(0, 2)
        x_6 = sb.atan(x)
        x_7 = sb.atan(x_6)
        x_8 = sb.atan(x_7)
        x_9 = sb.stream_barrier()
        x_10 = sb.set_stream(0, 0)
        x_11 = sb.make_tuple([x_1, x_4, x_8])
        x_12 = sb.concatenate(x_11, 0)
        return tvm.relay.Function([x], sb.ret(x_12))

    # We verify the correctness of the pass by structural_equal here, but it does not check the
    # equivalence of raf's extended constant. See issue #700.
    print(raf.ir.AsText(mod))
    assert tvm.ir.structural_equal(mod["main"], expected())


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_schedule_branch_in_branch():
    class Model(raf.Model):
        # wavefront schedule
        # wave 1
        #   chain 1: op 1
        #   chain 2: op 2, op 3
        #   chain 3: op 7, op 8, op 9
        # wave 2:
        #   chain 1: op 4
        #   chain 2: op 5
        # wave 3:
        #   chain 1: op 6, op 10
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)  # op 1

            p_1 = raf.atan(x)  # op 2
            p_1 = raf.atan(p_1)  # op 3
            p_1a = raf.atan(p_1)  # op 4
            p_1b = raf.atan(p_1)  # op 5
            p_1 = raf.concatenate([p_1a, p_1b])  # op 6

            p_2 = raf.atan(x)  # op 7
            p_2 = raf.atan(p_2)  # op 8
            p_2 = raf.atan(p_2)  # op 9
            return raf.concatenate([p_0, p_1, p_2])  # op 10

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(opt_level=2, config={"raf.stream_schedule.policy": "wavefront"}):
        mod = RAFSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
        def @main(%x: Tensor[(2, 2), float32]) {
          let %x_0 = raf.op.set_stream(int64(0), int64(0));
          let %x_1 = raf.op.atan(%x);
          let %x_2 = raf.op.set_stream(int64(0), int64(1));
          let %x_3 = raf.op.atan(%x);
          let %x_4 = raf.op.atan(%x_3);
          let %x_5 = raf.op.set_stream(int64(0), int64(2));
          let %x_6 = raf.op.atan(%x);
          let %x_7 = raf.op.atan(%x_6);
          let %x_8 = raf.op.atan(%x_7);
          let %x_9 = raf.op.stream_barrier();
          let %x_10 = raf.op.set_stream(int64(0), int64(0));
          let %x_11 = raf.op.atan(%x_4);
          let %x_12 = raf.op.set_stream(int64(0), int64(1));
          let %x_13 = raf.op.atan(%x_4);
          let %x_14 = raf.op.stream_barrier();
          let %x_15 = raf.op.set_stream(int64(0), int64(0));
          let %x_16 = (%x_11, %x_13);
          let %x_17 = raf.op.concatenate(%x_16, int64(0));
          let %x_18 = (%x_1, %x_17, %x_8);
          let %x_19 = raf.op.concatenate(%x_18, int64(0));
          %x_19
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)
        x_2 = sb.set_stream(0, 1)
        x_3 = sb.atan(x)
        x_4 = sb.atan(x_3)
        x_5 = sb.set_stream(0, 2)
        x_6 = sb.atan(x)
        x_7 = sb.atan(x_6)
        x_8 = sb.atan(x_7)
        x_9 = sb.stream_barrier()
        x_10 = sb.set_stream(0, 0)
        x_11 = sb.atan(x_4)
        x_12 = sb.set_stream(0, 1)
        x_13 = sb.atan(x_4)
        x_14 = sb.stream_barrier()
        x_15 = sb.set_stream(0, 0)
        x_16 = sb.make_tuple([x_11, x_13])
        x_17 = sb.concatenate(x_16, 0)
        x_18 = sb.make_tuple([x_1, x_17, x_8])
        x_19 = sb.concatenate(x_18, 0)
        return tvm.relay.Function([x], sb.ret(x_19))

    assert tvm.ir.structural_equal(mod["main"], expected())


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_schedule_stacked_blocks():
    class Model(raf.Model):
        # wavefront schedule
        # wave 1
        #   chain 1: op 1
        #   chain 2: op 2,
        #   chain 3: op 3, op 4
        # wave 2:
        #   chain 1: op 5
        # wave 3:
        #   chain 1: op 6
        #   chain 2: op 7
        #   chain 3: op 8, op 9
        # wave 4:
        #   chain 1: op 10
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)  # op 1
            p_1 = raf.atan(x)  # op 2
            p_2 = raf.atan(x)  # op 3
            p_2 = raf.atan(p_2)  # op 4
            x = raf.concatenate([p_0, p_1, p_2])  # op 5
            p_0 = raf.atan(x)  # op 6
            p_1 = raf.atan(x)  # op 7
            p_2 = raf.atan(x)  # op 8
            p_2 = raf.atan(p_2)  # op 9
            return raf.concatenate([p_0, p_1, p_2])  # op 10

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(opt_level=2, config={"raf.stream_schedule.policy": "wavefront"}):
        mod = RAFSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
        def @main(%x: Tensor[(2, 2), float32]) {
          let %x_0 = raf.op.set_stream(int64(0), int64(0));
          let %x_1 = raf.op.atan(%x);
          let %x_2 = raf.op.set_stream(int64(0), int64(1));
          let %x_3 = raf.op.atan(%x);
          let %x_4 = raf.op.set_stream(int64(0), int64(2));
          let %x_5 = raf.op.atan(%x);
          let %x_6 = raf.op.atan(%x_5);
          let %x_7 = raf.op.stream_barrier();
          let %x_8 = raf.op.set_stream(int64(0), int64(0));
          let %x_9 = (%x_1, %x_3, %x_6);
          let %x_10 = raf.op.concatenate(%x_9, int64(0));
          let %x_11 = raf.op.stream_barrier();
          let %x_12 = raf.op.set_stream(int64(0), int64(0));
          let %x_13 = raf.op.atan(%x_10);
          let %x_14 = raf.op.set_stream(int64(0), int64(1));
          let %x_15 = raf.op.atan(%x_10);
          let %x_16 = raf.op.set_stream(int64(0), int64(2));
          let %x_17 = raf.op.atan(%x_10);
          let %x_18 = raf.op.atan(%x_17);
          let %x_19 = raf.op.stream_barrier();
          let %x_20 = raf.op.set_stream(int64(0), int64(0));
          let %x_21 = (%x_13, %x_15, %x_18);
          let %x_22 = raf.op.concatenate(%x_21, int64(0));
          %x_22
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 0)
        x_1 = sb.atan(x)
        x_2 = sb.set_stream(0, 1)
        x_3 = sb.atan(x)
        x_4 = sb.set_stream(0, 2)
        x_5 = sb.atan(x)
        x_6 = sb.atan(x_5)
        x_7 = sb.stream_barrier()
        x_8 = sb.set_stream(0, 0)
        x_9 = sb.make_tuple([x_1, x_3, x_6])
        x_10 = sb.concatenate(x_9, 0)
        x_11 = sb.stream_barrier()
        x_12 = sb.set_stream(0, 0)
        x_13 = sb.atan(x_10)
        x_14 = sb.set_stream(0, 1)
        x_15 = sb.atan(x_10)
        x_16 = sb.set_stream(0, 2)
        x_17 = sb.atan(x_10)
        x_18 = sb.atan(x_17)
        x_19 = sb.stream_barrier()
        x_20 = sb.set_stream(0, 0)
        x_21 = sb.make_tuple([x_13, x_15, x_18])
        x_22 = sb.concatenate(x_21, 0)
        return tvm.relay.Function([x], sb.ret(x_22))

    assert tvm.ir.structural_equal(mod["main"], expected())


if __name__ == "__main__":
    pytest.main([__file__])
