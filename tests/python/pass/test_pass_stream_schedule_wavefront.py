# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
from typing import Dict, List
import pytest
import tvm
import tvm.relay
import mnm
from mnm.testing import randn
from mnm.ir.pass_manager import MNMSequential
from mnm._ffi.pass_ import ToGraphNormalForm, WavefrontStreamSchedule
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
        return self.scope_builder.let('', tvm.relay.Tuple(fields))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let('', tvm.relay.Call(self.get_operator(op_name), args))

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
def test_wavefront_schedule_three_simple_branches():
    class Model(mnm.Model):
        # wavefront schedule:
        # wave 1
        #   chain 1: op 1
        #   chain 2: op 2, op 3
        #   chain 3: op 4, op 5, op 6
        # wave 2
        #   chain 1: op 7
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)   # op 1

            p_1 = mnm.atan(x)   # op 2
            p_1 = mnm.atan(p_1)  # op 3

            p_2 = mnm.atan(x)   # op 4
            p_2 = mnm.atan(p_2)  # op 5
            p_2 = mnm.atan(p_2)  # op 6
            return mnm.concatenate([p_0, p_1, p_2])  # op 7
    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(opt_level=2, config={"mnm.stream_schedule.policy": 'wavefront'}):
        mod = MNMSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
         #[version = "0.0.5"]
         fn (%x: Tensor[(2, 2), float32]) {
           let %x_0 = mnm.op.set_stream(int64(0), int64(1));
           let %x_1 = mnm.op.atan(%x);
           let %x_2 = mnm.op.add_event(int64(0));
           let %x_3 = mnm.op.set_stream(int64(0), int64(2));
           let %x_4 = mnm.op.atan(%x);
           let %x_5 = mnm.op.atan(%x_4);
           let %x_6 = mnm.op.add_event(int64(1));
           let %x_7 = mnm.op.set_stream(int64(0), int64(3));
           let %x_8 = mnm.op.atan(%x);
           let %x_9 = mnm.op.atan(%x_8);
           let %x_10 = mnm.op.atan(%x_9);
           let %x_11 = mnm.op.add_event(int64(2));
           let %x_12 = mnm.op.set_stream(int64(0), int64(0));
           let %x_13 = mnm.op.wait_event(int64(0));
           let %x_14 = mnm.op.wait_event(int64(1));
           let %x_15 = mnm.op.wait_event(int64(2));
           let %x_16 = mnm.op.add_event(int64(3));
           let %x_17 = mnm.op.set_stream(int64(0), int64(1));
           let %x_18 = mnm.op.wait_event(int64(3));
           let %x_19 = (%x_1, %x_5, %x_10);
           let %x_20 = mnm.op.concatenate(%x_19, int64(0));
           %x_21
         }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)
        x_2 = sb.add_event(0)
        x_3 = sb.set_stream(0, 2)
        x_4 = sb.atan(x)
        x_5 = sb.atan(x_4)
        x_6 = sb.add_event(1)
        x_7 = sb.set_stream(0, 3)
        x_8 = sb.atan(x)
        x_9 = sb.atan(x_8)
        x_10 = sb.atan(x_9)
        x_11 = sb.add_event(2)
        x_12 = sb.set_stream(0, 0)
        x_13 = sb.wait_event(0)
        x_14 = sb.wait_event(1)
        x_15 = sb.wait_event(2)
        x_16 = sb.add_event(3)
        x_17 = sb.set_stream(0, 1)
        x_18 = sb.wait_event(3)
        x_19 = sb.make_tuple([x_1, x_5, x_10])
        x_20 = sb.concatenate(x_19, 0)
        return tvm.relay.Function([x], sb.ret(x_20))

    # We verify the correctness of the pass by structural_equal here, but it does not check the
    # equivalence of meta's extended constant. See issue #700.
    assert tvm.ir.structural_equal(mod['main'], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_schedule_branch_in_branch():
    class Model(mnm.Model):
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

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)    # op 1

            p_1 = mnm.atan(x)    # op 2
            p_1 = mnm.atan(p_1)   # op 3
            p_1a = mnm.atan(p_1)  # op 4
            p_1b = mnm.atan(p_1)  # op 5
            p_1 = mnm.concatenate([p_1a, p_1b])    # op 6

            p_2 = mnm.atan(x)    # op 7
            p_2 = mnm.atan(p_2)   # op 8
            p_2 = mnm.atan(p_2)   # op 9
            return mnm.concatenate([p_0, p_1, p_2])    # op 10
    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(opt_level=2, config={"mnm.stream_schedule.policy": 'wavefront'}):
        mod = MNMSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
         #[version = "0.0.5"]
         fn (%x: Tensor[(2, 2), float32]) {
           let %x_0 = mnm.op.set_stream(int64(0), int64(1));
           let %x_1 = mnm.op.atan(%x);
           let %x_2 = mnm.op.add_event(int64(0));
           let %x_3 = mnm.op.set_stream(int64(0), int64(2));
           let %x_4 = mnm.op.atan(%x);
           let %x_5 = mnm.op.atan(%x_4);
           let %x_6 = mnm.op.add_event(int64(1));
           let %x_7 = mnm.op.set_stream(int64(0), int64(3));
           let %x_8 = mnm.op.atan(%x);
           let %x_9 = mnm.op.atan(%x_8);
           let %x_10 = mnm.op.atan(%x_9);
           let %x_11 = mnm.op.add_event(int64(2));
           let %x_12 = mnm.op.set_stream(int64(0), int64(0));
           let %x_13 = mnm.op.wait_event(int64(0));
           let %x_14 = mnm.op.wait_event(int64(1));
           let %x_15 = mnm.op.wait_event(int64(2));
           let %x_16 = mnm.op.add_event(int64(3));
           let %x_17 = mnm.op.set_stream(int64(0), int64(1));
           let %x_18 = mnm.op.wait_event(int64(3));
           let %x_19 = mnm.op.atan(%x_5);
           let %x_20 = mnm.op.add_event(int64(4));
           let %x_21 = mnm.op.set_stream(int64(0), int64(2));
           let %x_22 = mnm.op.wait_event(int64(3));
           let %x_23 = mnm.op.atan(%x_5);
           let %x_24 = mnm.op.add_event(int64(5));
           let %x_25 = mnm.op.set_stream(int64(0), int64(0));
           let %x_26 = mnm.op.wait_event(int64(4));
           let %x_27 = mnm.op.wait_event(int64(5));
           let %x_28 = mnm.op.add_event(int64(6));
           let %x_29 = mnm.op.set_stream(int64(0), int64(1));
           let %x_30 = mnm.op.wait_event(int64(6));
           let %x_31 = (%x_19, %x_23);
           let %x_32 = mnm.op.concatenate(%x_31, int64(0));
           let %x_33 = (%x_1, %x_32, %x_10);
           let %x_34 = mnm.op.concatenate(%x_33, int64(0));
           %x_36
         }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)
        x_2 = sb.add_event(0)
        x_3 = sb.set_stream(0, 2)
        x_4 = sb.atan(x)
        x_5 = sb.atan(x_4)
        x_6 = sb.add_event(1)
        x_7 = sb.set_stream(0, 3)
        x_8 = sb.atan(x)
        x_9 = sb.atan(x_8)
        x_10 = sb.atan(x_9)
        x_11 = sb.add_event(2)
        x_12 = sb.set_stream(0, 0)
        x_13 = sb.wait_event(0)
        x_14 = sb.wait_event(1)
        x_15 = sb.wait_event(2)
        x_16 = sb.add_event(3)
        x_17 = sb.set_stream(0, 1)
        x_18 = sb.wait_event(3)
        x_19 = sb.atan(x_5)
        x_20 = sb.add_event(4)
        x_21 = sb.set_stream(0, 2)
        x_22 = sb.wait_event(3)
        x_23 = sb.atan(x_5)
        x_24 = sb.add_event(5)
        x_25 = sb.set_stream(0, 0)
        x_26 = sb.wait_event(4)
        x_27 = sb.wait_event(5)
        x_28 = sb.add_event(6)
        x_29 = sb.set_stream(0, 1)
        x_30 = sb.wait_event(6)
        x_31 = sb.make_tuple([x_19, x_23])
        x_32 = sb.concatenate(x_31, 0)
        x_33 = sb.make_tuple([x_1, x_32, x_10])
        x_34 = sb.concatenate(x_33, 0)
        return tvm.relay.Function([x], sb.ret(x_34))

    assert tvm.ir.structural_equal(mod['main'], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_schedule_stacked_blocks():
    class Model(mnm.Model):
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

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)    # op 1
            p_1 = mnm.atan(x)    # op 2
            p_2 = mnm.atan(x)    # op 3
            p_2 = mnm.atan(p_2)  # op 4
            x = mnm.concatenate([p_0, p_1, p_2])    # op 5
            p_0 = mnm.atan(x)    # op 6
            p_1 = mnm.atan(x)    # op 7
            p_2 = mnm.atan(x)    # op 8
            p_2 = mnm.atan(p_2)  # op 9
            return mnm.concatenate([p_0, p_1, p_2])    # op 10
    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(opt_level=2, config={"mnm.stream_schedule.policy": 'wavefront'}):
        mod = MNMSequential([ToGraphNormalForm(), WavefrontStreamSchedule()])(mod)

    def expected():
        """
        fn (%x: Tensor[(2, 2), float32]) {
          let %x_0 = mnm.op.set_stream(int64(0), int64(1));
          let %x_1 = mnm.op.atan(%x);
          let %x_2 = mnm.op.add_event(int64(0));
          let %x_3 = mnm.op.set_stream(int64(0), int64(2));
          let %x_4 = mnm.op.atan(%x);
          let %x_5 = mnm.op.add_event(int64(1));
          let %x_6 = mnm.op.set_stream(int64(0), int64(3));
          let %x_7 = mnm.op.atan(%x);
          let %x_8 = mnm.op.atan(%x_7);
          let %x_9 = mnm.op.add_event(int64(2));
          let %x_10 = mnm.op.set_stream(int64(0), int64(0));
          let %x_11 = mnm.op.wait_event(int64(0));
          let %x_12 = mnm.op.wait_event(int64(1));
          let %x_13 = mnm.op.wait_event(int64(2));
          let %x_14 = mnm.op.add_event(int64(3));
          let %x_15 = mnm.op.set_stream(int64(0), int64(1));
          let %x_16 = mnm.op.wait_event(int64(3));
          let %x_17 = (%x_1, %x_4, %x_8);
          let %x_18 = mnm.op.concatenate(%x_17, int64(0));
          let %x_19 = mnm.op.add_event(int64(4));
          let %x_20 = mnm.op.set_stream(int64(0), int64(0));
          let %x_21 = mnm.op.wait_event(int64(4));
          let %x_22 = mnm.op.add_event(int64(5));
          let %x_23 = mnm.op.set_stream(int64(0), int64(1));
          let %x_24 = mnm.op.wait_event(int64(5));
          let %x_25 = mnm.op.atan(%x_18);
          let %x_26 = mnm.op.add_event(int64(6));
          let %x_27 = mnm.op.set_stream(int64(0), int64(2));
          let %x_28 = mnm.op.wait_event(int64(5));
          let %x_29 = mnm.op.atan(%x_18);
          let %x_30 = mnm.op.add_event(int64(7));
          let %x_31 = mnm.op.set_stream(int64(0), int64(3));
          let %x_32 = mnm.op.wait_event(int64(5));
          let %x_33 = mnm.op.atan(%x_18);
          let %x_34 = mnm.op.atan(%x_33);
          let %x_35 = mnm.op.add_event(int64(8));
          let %x_36 = mnm.op.set_stream(int64(0), int64(0));
          let %x_37 = mnm.op.wait_event(int64(6));
          let %x_38 = mnm.op.wait_event(int64(7));
          let %x_39 = mnm.op.wait_event(int64(8));
          let %x_40 = mnm.op.add_event(int64(9));
          let %x_41 = mnm.op.set_stream(int64(0), int64(1));
          let %x_42 = mnm.op.wait_event(int64(9));
          let %x_43 = (%x_25, %x_29, %x_34);
          let %x_44 = mnm.op.concatenate(%x_43, int64(0));
          %x_44
        }
        """
        sb = ANFBuilder()
        x = extended_var("x", shape=input_shape)
        x_0 = sb.set_stream(0, 1)
        x_1 = sb.atan(x)
        x_2 = sb.add_event(0)
        x_3 = sb.set_stream(0, 2)
        x_4 = sb.atan(x)
        x_5 = sb.add_event(1)
        x_6 = sb.set_stream(0, 3)
        x_7 = sb.atan(x)
        x_8 = sb.atan(x_7)
        x_9 = sb.add_event(2)
        x_10 = sb.set_stream(0, 0)
        x_11 = sb.wait_event(0)
        x_12 = sb.wait_event(1)
        x_13 = sb.wait_event(2)
        x_14 = sb.add_event(3)
        x_15 = sb.set_stream(0, 1)
        x_16 = sb.wait_event(3)
        x_17 = sb.make_tuple([x_1, x_4, x_8])
        x_18 = sb.concatenate(x_17, 0)
        x_19 = sb.add_event(4)
        x_20 = sb.set_stream(0, 0)
        x_21 = sb.wait_event(4)
        x_22 = sb.add_event(5)
        x_23 = sb.set_stream(0, 1)
        x_24 = sb.wait_event(5)
        x_25 = sb.atan(x_18)
        x_26 = sb.add_event(6)
        x_27 = sb.set_stream(0, 2)
        x_28 = sb.wait_event(5)
        x_29 = sb.atan(x_18)
        x_30 = sb.add_event(7)
        x_31 = sb.set_stream(0, 3)
        x_32 = sb.wait_event(5)
        x_33 = sb.atan(x_18)
        x_34 = sb.atan(x_33)
        x_35 = sb.add_event(8)
        x_36 = sb.set_stream(0, 0)
        x_37 = sb.wait_event(6)
        x_38 = sb.wait_event(7)
        x_39 = sb.wait_event(8)
        x_40 = sb.add_event(9)
        x_41 = sb.set_stream(0, 1)
        x_42 = sb.wait_event(9)
        x_43 = sb.make_tuple([x_25, x_29, x_34])
        x_44 = sb.concatenate(x_43, 0)

        return tvm.relay.Function([x], sb.ret(x_44))

    assert tvm.ir.structural_equal(mod['main'], expected())


if __name__ == '__main__':
    pytest.main([__file__])
