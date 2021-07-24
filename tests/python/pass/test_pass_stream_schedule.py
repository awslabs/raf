# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
import pytest
import tvm
import tvm.relay
import mnm
from mnm.testing import randn
from mnm.ir.pass_manager import MNMSequential
from mnm._ffi.pass_ import StreamSchedule, ToGraphNormalForm
from mnm._core.ir_ext import extended_var


class LetBuilder:
    def __init__(self):
        self.vars = []
        self.values = []

    def append(self, value):
        self.vars.append(extended_var(name_hint=""))
        self.values.append(value)
        return self.vars[-1]

    def ret(self, body):
        for var, value in reversed(list(zip(self.vars, self.values))):
            body = tvm.relay.Let(var, value, body)
        self.vars.clear()
        self.values.clear()
        return body


LET_BUILDER = LetBuilder()


def const(value):
    return LET_BUILDER.append(mnm.ir.const(value))


def make_tuple(fields):
    return LET_BUILDER.append(tvm.relay.Tuple(fields))


def call(op_name, args):
    op = tvm.relay.op.get(f"mnm.op.{op_name}")
    return LET_BUILDER.append(tvm.relay.Call(op, args))


def set_stream(device_id: int, stream_id: int):
    device_id = mnm.ir.const(device_id)
    stream_id = mnm.ir.const(stream_id)
    return call("set_stream", [device_id, stream_id])


def add_event(event_id: int):
    event_id = mnm.ir.const(event_id)
    return call("add_event", [event_id])


def wait_event(event_id: int):
    event_id = mnm.ir.const(event_id)
    return call("wait_event", [event_id])


def atan(x: tvm.ir.RelayExpr):
    return call("atan", [x])


def concatenate(x: tvm.ir.RelayExpr, axis: tvm.ir.RelayExpr):
    return call("concatenate", [x, axis])


def ret(body: tvm.ir.RelayExpr):
    return LET_BUILDER.ret(body)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_three_simple_branches():
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
        mod = MNMSequential([ToGraphNormalForm(), StreamSchedule()])(mod)

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
           let %x_20 = int64(0);
           let %x_21 = mnm.op.concatenate(%x_19, %x_20);
           %x_21
         }
        """
        x = extended_var("x", shape=input_shape)
        x_0 = set_stream(0, 1)
        x_1 = atan(x)
        x_2 = add_event(0)
        x_3 = set_stream(0, 2)
        x_4 = atan(x)
        x_5 = atan(x_4)
        x_6 = add_event(1)
        x_7 = set_stream(0, 3)
        x_8 = atan(x)
        x_9 = atan(x_8)
        x_10 = atan(x_9)
        x_11 = add_event(2)
        x_12 = set_stream(0, 0)
        x_13 = wait_event(0)
        x_14 = wait_event(1)
        x_15 = wait_event(2)
        x_16 = add_event(3)
        x_17 = set_stream(0, 1)
        x_18 = wait_event(3)
        x_19 = make_tuple([x_1, x_5, x_10])
        x_20 = const(0)
        x_21 = concatenate(x_19, x_20)
        return tvm.relay.Function([x], ret(x_21))

    # We verify the correctness of the pass by structural_equal here, but it does not check the
    # equivalence of meta's extended constant. This is an existing issue and would be fix in the
    # future.
    assert tvm.ir.structural_equal(mod['main'], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_branch_in_branch():
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
        mod = MNMSequential([ToGraphNormalForm(), StreamSchedule()])(mod)

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
           let %x_32 = int64(0);
           let %x_33 = mnm.op.concatenate(%x_31, %x_32);
           let %x_34 = (%x_1, %x_33, %x_10);
           let %x_35 = int64(0);
           let %x_36 = mnm.op.concatenate(%x_34, %x_35);
           %x_36
         }
        """
        x = extended_var("x", shape=input_shape)
        x_0 = set_stream(0, 1)
        x_1 = atan(x)
        x_2 = add_event(0)
        x_3 = set_stream(0, 2)
        x_4 = atan(x)
        x_5 = atan(x_4)
        x_6 = add_event(1)
        x_7 = set_stream(0, 3)
        x_8 = atan(x)
        x_9 = atan(x_8)
        x_10 = atan(x_9)
        x_11 = add_event(2)
        x_12 = set_stream(0, 0)
        x_13 = wait_event(0)
        x_14 = wait_event(1)
        x_15 = wait_event(2)
        x_16 = add_event(3)
        x_17 = set_stream(0, 1)
        x_18 = wait_event(3)
        x_19 = atan(x_5)
        x_20 = add_event(4)
        x_21 = set_stream(0, 2)
        x_22 = wait_event(3)
        x_23 = atan(x_5)
        x_24 = add_event(5)
        x_25 = set_stream(0, 0)
        x_26 = wait_event(4)
        x_27 = wait_event(5)
        x_28 = add_event(6)
        x_29 = set_stream(0, 1)
        x_30 = wait_event(6)
        x_31 = make_tuple([x_19, x_23])
        x_32 = const(0)
        x_33 = concatenate(x_31, x_32)
        x_34 = make_tuple([x_1, x_33, x_10])
        x_35 = const(0)
        x_36 = concatenate(x_34, x_35)
        return tvm.relay.Function([x], ret(x_36))

    assert tvm.ir.structural_equal(mod['main'], expected())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_wavefront_stacked_blocks():
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
        mod = MNMSequential([ToGraphNormalForm(), StreamSchedule()])(mod)

    def expected():
        """
         #[version = "0.0.5"]
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
           let %x_18 = int64(0);
           let %x_19 = mnm.op.concatenate(%x_17, %x_18);
           let %x_20 = mnm.op.add_event(int64(4));
           let %x_21 = mnm.op.set_stream(int64(0), int64(0));
           let %x_22 = mnm.op.wait_event(int64(4));
           let %x_23 = mnm.op.add_event(int64(5));
           let %x_24 = mnm.op.set_stream(int64(0), int64(1));
           let %x_25 = mnm.op.wait_event(int64(5));
           let %x_26 = mnm.op.atan(%x_19);
           let %x_27 = mnm.op.add_event(int64(6));
           let %x_28 = mnm.op.set_stream(int64(0), int64(2));
           let %x_29 = mnm.op.wait_event(int64(5));
           let %x_30 = mnm.op.atan(%x_19);
           let %x_31 = mnm.op.add_event(int64(7));
           let %x_32 = mnm.op.set_stream(int64(0), int64(3));
           let %x_33 = mnm.op.wait_event(int64(5));
           let %x_34 = mnm.op.atan(%x_19);
           let %x_35 = mnm.op.atan(%x_34);
           let %x_36 = mnm.op.add_event(int64(8));
           let %x_37 = mnm.op.set_stream(int64(0), int64(0));
           let %x_38 = mnm.op.wait_event(int64(6));
           let %x_39 = mnm.op.wait_event(int64(7));
           let %x_40 = mnm.op.wait_event(int64(8));
           let %x_41 = mnm.op.add_event(int64(9));
           let %x_42 = mnm.op.set_stream(int64(0), int64(1));
           let %x_43 = mnm.op.wait_event(int64(9));
           let %x_44 = (%x_26, %x_30, %x_35);
           let %x_45 = int64(0);
           let %x_46 = mnm.op.concatenate(%x_44, %x_45);
           %x_46
         }
        """
        x = extended_var("x", shape=input_shape)
        x_0 = set_stream(0, 1)
        x_1 = atan(x)
        x_2 = add_event(0)
        x_3 = set_stream(0, 2)
        x_4 = atan(x)
        x_5 = add_event(1)
        x_6 = set_stream(0, 3)
        x_7 = atan(x)
        x_8 = atan(x_7)
        x_9 = add_event(2)
        x_10 = set_stream(0, 0)
        x_11 = wait_event(0)
        x_12 = wait_event(1)
        x_13 = wait_event(2)
        x_14 = add_event(3)
        x_15 = set_stream(0, 1)
        x_16 = wait_event(3)
        x_17 = make_tuple([x_1, x_4, x_8])
        x_18 = const(0)
        x_19 = concatenate(x_17, x_18)
        x_20 = add_event(4)
        x_21 = set_stream(0, 0)
        x_22 = wait_event(4)
        x_23 = add_event(5)
        x_24 = set_stream(0, 1)
        x_25 = wait_event(5)
        x_26 = atan(x_19)
        x_27 = add_event(6)
        x_28 = set_stream(0, 2)
        x_29 = wait_event(5)
        x_30 = atan(x_19)
        x_31 = add_event(7)
        x_32 = set_stream(0, 3)
        x_33 = wait_event(5)
        x_34 = atan(x_19)
        x_35 = atan(x_34)
        x_36 = add_event(8)
        x_37 = set_stream(0, 0)
        x_38 = wait_event(6)
        x_39 = wait_event(7)
        x_40 = wait_event(8)
        x_41 = add_event(9)
        x_42 = set_stream(0, 1)
        x_43 = wait_event(9)
        x_44 = make_tuple([x_26, x_30, x_35])
        x_45 = const(0)
        x_46 = concatenate(x_44, x_45)
        return tvm.relay.Function([x], ret(x_46))

    assert tvm.ir.structural_equal(mod['main'], expected())


if __name__ == '__main__':
    pytest.main([__file__])
