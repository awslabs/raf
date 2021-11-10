# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use
from typing import Dict, List
import pytest
import tvm

import mnm
from mnm._core.device import Device
from mnm import distributed as dist
from mnm.ir.pass_manager import MNMSequential
from mnm._ffi.pass_ import AnnotateDistOps
from mnm.testing import randn
from mnm._core.ir_ext import extended_var
from mnm.ir import ScopeBuilder

class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators: Dict[str, tvm.ir.Op] = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = mnm._ffi.op.GetOp(f"mnm.op.{op_name}")
        return self.operators[op_name]

    def const(self, value):
        return mnm.ir.const(value)

    def make_tuple(self, fields):
        return self.scope_builder.let('', tvm.relay.Tuple(fields))

    def get_tuple_item(self, tup, index):
        return self.scope_builder.let('', tvm.relay.TupleGetItem(tup, index))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let('', tvm.relay.Call(self.get_operator(op_name), args))

    def set_stream(self, device_id: int, stream_id: int):
        device_id = mnm.ir.const(device_id)
        stream_id = mnm.ir.const(stream_id)
        return self.call("set_stream", [device_id, stream_id])

    def add_event(self, event_id: int, stream_id: int):
        event_id = mnm.ir.const(event_id)
        stream_id = mnm.ir.const(stream_id)
        return self.call("add_event", [event_id, stream_id])

    def wait_event(self, event_id: int, stream_id: int):
        event_id = mnm.ir.const(event_id)
        stream_id = mnm.ir.const(stream_id)
        return self.call("wait_event", [event_id, stream_id])

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_single_allreduce(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        class Model(mnm.Model):
            # mul -> allreduce -> mul
            def build(self):
                pass

            @mnm.model.trace
            def forward(self, x):
                a0 = mnm.atan(x)
                a1 = mnm.allreduce(a0)
                a2 = mnm.atan(a1)
                return a2

        model = Model()
        x, _ = randn(shape)
        mod = model._internal(x).mod

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %a1 = mnm.op.atan(%x);
                let %a2 = (%a1,);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %a3 = mnm.op._allreduce(%a2, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
                let %set_stream_comp2 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
                let %a4 = mnm.op.atan(%a3);
                %a4
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            builder.set_stream(0, comp_stream)
            x_1 = builder.call("atan", [x])
            x_2 = builder.make_tuple([x_1])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_2 = builder.call("_allreduce", [x_2, mnm.ir.const("sum")])
            builder.add_event(2, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(2, comp_stream)
            x_3 = builder.call("atan", [x_2])
            return tvm.relay.Function([x], builder.ret(x_3))
    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_parallel_allreduce(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        class Model(mnm.Model):
            #    /-> atan -> allreduce -> atan -\
            # atan                               concat
            #    \-> relu -> allreduce -> atan -/
            def build(self):
                pass

            @mnm.model.trace
            def forward(self, x):
                a0 = mnm.atan(x)
                a1_a = mnm.atan(a0)
                a1_b = mnm.relu(a0)
                a2_a = mnm.allreduce(a1_a)
                a2_b = mnm.allreduce(a1_b)
                a3_a = mnm.atan(a2_a)
                a3_b = mnm.atan(a2_b)
                a4 = mnm.concatenate([a3_a, a3_b])
                return a4

        model = Model()
        x, _ = randn(shape)
        mod = model._internal(x).mod

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %a1 = mnm.op.atan(%x);
                let %a2 = mnm.op.atan(%a1);
                let %a3 = (%a2,);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %a4 = mnm.op._allreduce(%a3, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(3), int64(5));
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(3), int64(1));
                let %a5 = mnm.op.atan(%a4);
                let %a6 = mnm.op.relu(%a1);
                let %a7 = (%a6,);
                let %add_event_comp1 = mnm.op.add_event(int64(2), int64(1));
                let %set_stream_comm1 = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp1 = mnm.op.wait_event(int64(2), int64(5));
                let %a8 = mnm.op._allreduce(%a7, str"sum");
                let %add_event_comm1 = mnm.op.add_event(int64(4), int64(5));
                let %set_stream_comp2 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm1 = mnm.op.wait_event(int64(4), int64(1));
                let %a9 = mnm.op.atan(%a8);
                let %a10 = (%a5, %a9);
                let %a11 = mnm.op.concatenate(%a10, int64(0));
                %a11
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)

            builder.set_stream(0, comp_stream)
            x_1 = builder.call("atan", [x])

            x_2 = builder.call("atan", [x_1])
            x_3i = builder.make_tuple([x_2])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_3 = builder.call("_allreduce", [x_3i, mnm.ir.const("sum")])
            builder.add_event(2, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(2, comp_stream)
            x_4 = builder.call("atan", [x_3])

            x_5 = builder.call("relu", [x_1])
            x_6i = builder.make_tuple([x_5])
            builder.add_event(3, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(3, comm_stream)
            x_6 = builder.call("_allreduce", [x_6i, mnm.ir.const("sum")])
            builder.add_event(4, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(4, comp_stream)
            x_7 = builder.call("atan", [x_6])

            x_8i = builder.make_tuple([x_4, x_7])
            x_8 = builder.call("concatenate", [x_8i, mnm.ir.const(0)])
            return tvm.relay.Function([x], builder.ret(x_8))

    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_redundant_comm_to_comp_sync(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        def construct_model_func():
            # order in ANF is shown in parenthesis, synchronization (2)->(6) is not needed
            # atan (1) -> atan (3) -> allreduce (4) -> mul (5) -> concat (6)
            #    \                                           /
            #     -------->  allreduce (2) ---------------->
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            x_1 = builder.call("atan", [x])
            x_2i = builder.make_tuple([x_1])
            x_2 = builder.call("_allreduce", [x_2i, mnm.ir.const("sum")])
            x_3 = builder.call("atan", [x_1])
            x_4i = builder.make_tuple([x_3])
            x_4 = builder.call("_allreduce", [x_4i, mnm.ir.const("sum")])
            x_5 = builder.call("atan", [x_4])
            x_6i = builder.make_tuple([x_5, x_2])
            x_6 = builder.call("concatenate", [x_6i, mnm.ir.const(0)])
            return tvm.relay.Function([x], builder.ret(x_6))

        mod = tvm.IRModule()
        mod["main"] = construct_model_func()

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %v = mnm.op.atan(%x);
                let %v1 = (%v,);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %v2 = mnm.op._allreduce(%v1, str"sum");
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %v3 = mnm.op.atan(%v);
                let %v4 = (%v3,);
                let %add_event_comp1 = mnm.op.add_event(int64(2), int64(1));
                let %set_stream_comm1 = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp1 = mnm.op.wait_event(int64(2), int64(5));
                let %v5 = mnm.op._allreduce(%v4, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(3), int64(5));
                let %set_stream_comp2 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(3), int64(1));
                let %v6 = mnm.op.atan(%v5);
                let %v7 = (%v6, %v2);
                let %v8 = mnm.op.concatenate(%v7, int64(0));
                %v8
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            builder.set_stream(0, comp_stream)
            x_1 = builder.call("atan", [x])
            x_2i = builder.make_tuple([x_1])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_2 = builder.call("_allreduce", [x_2i, mnm.ir.const("sum")])
            builder.set_stream(0, comp_stream)
            x_3 = builder.call("atan", [x_1])
            x_4i = builder.make_tuple([x_3])
            builder.add_event(2, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(2, comm_stream)
            x_4 = builder.call("_allreduce", [x_4i, mnm.ir.const("sum")])
            builder.add_event(3, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(3, comp_stream)
            x_5 = builder.call("atan", [x_4])

            x_6i = builder.make_tuple([x_5, x_2])
            x_6 = builder.call("concatenate", [x_6i, mnm.ir.const(0)])
            return tvm.relay.Function([x], builder.ret(x_6))

    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_redundant_comp_to_comm_sync(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        def construct_model_func():
            # order in ANF is shown in parenthesis, synchronization (1)->(4) is not needed
            # atan (1) -> atan (2) -> allreduce (3) -> atan (5) -> concat (6)
            #    \                                           /
            #     -------->  allreduce (4) ---------------->
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            x_1 = builder.call("atan", [x])
            x_4i = builder.make_tuple([x_1])
            x_2 = builder.call("atan", [x_1])
            x_3i = builder.make_tuple([x_2])
            x_3 = builder.call("_allreduce", [x_3i, mnm.ir.const("sum")])
            x_4 = builder.call("_allreduce", [x_4i, mnm.ir.const("sum")])
            x_5 = builder.call("atan", [x_3])
            x_6i = builder.make_tuple([x_5, x_4])
            x_6 = builder.call("concatenate", [x_6i, mnm.ir.const(0)])
            return tvm.relay.Function([x], builder.ret(x_6))

        mod = tvm.IRModule()
        mod["main"] = construct_model_func()

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %v = mnm.op.atan(%x);
                let %v1 = (%v,);
                let %v2 = mnm.op.atan(%v);
                let %v3 = (%v2,);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %v4 = mnm.op._allreduce(%v3, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
                let %v5 = mnm.op._allreduce(%v1, str"sum");
                let %add_event_comm1 = mnm.op.add_event(int64(3), int64(5));
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
                let %v6 = mnm.op.atan(%v4);
                let %wait_for_comm1 = mnm.op.wait_event(int64(3), int64(1));
                let %v7 = (%v6, %v5);
                let %v8 = mnm.op.concatenate(%v7, int64(0));
                %v8
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            builder.set_stream(0, comp_stream)
            x_1 = builder.call("atan", [x])
            x_4i = builder.make_tuple([x_1])
            x_2 = builder.call("atan", [x_1])
            x_3i = builder.make_tuple([x_2])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_3 = builder.call("_allreduce", [x_3i, mnm.ir.const("sum")])
            builder.add_event(2, comm_stream)
            x_4 = builder.call("_allreduce", [x_4i, mnm.ir.const("sum")])
            builder.add_event(3, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(3, comp_stream)
            x_5 = builder.call("atan", [x_3])
            builder.wait_event(2, comp_stream)

            x_6i = builder.make_tuple([x_5, x_4])
            x_6 = builder.call("concatenate", [x_6i, mnm.ir.const(0)])
            return tvm.relay.Function([x], builder.ret(x_6))

    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_multi_input_allreduce(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        class Model(mnm.Model):
            # x -> atan -->allreduce -> mul
            #    \         /
            #     -> atan->
            def build(self):
                pass

            @mnm.model.trace
            def forward(self, x):
                a1_a = mnm.atan(x)
                a1_b = mnm.atan(x)
                b = mnm.allreduce([a1_a, a1_b])
                c = mnm.multiply(b[0], b[1])
                return c

        model = Model()
        x, _ = randn(shape)
        mod = model._internal(x).mod

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %a1 = mnm.op.atan(%x);
                let %a2 = mnm.op.atan(%x);
                let %a3 = (%a1, %a2);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %a4 = mnm.op._allreduce(%a3, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
                let %a5 = %a4.0;
                let %a6 = %a4.1;
                let %a7 = mnm.op.multiply(%a5, %a6);
                %a7
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            builder.set_stream(0, comp_stream)

            x_1_a = builder.call("atan", [x])
            x_1_b = builder.call("atan", [x])
            x_2 = builder.make_tuple([x_1_a, x_1_b])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_3 = builder.call("_allreduce", [x_2, mnm.ir.const("sum")])
            builder.add_event(2, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(2, comp_stream)
            x_4_a = builder.get_tuple_item(x_3, 0)
            x_4_b = builder.get_tuple_item(x_3, 1)
            x_5 = builder.call("multiply", [x_4_a, x_4_b])

            return tvm.relay.Function([x], builder.ret(x_5))

    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape,comp_stream,comm_stream", [
    [(64, 128), 1, 5]
])
def test_multi_user_allreduce(shape, comp_stream, comm_stream):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True

    with Device("cuda(0)"):
        class Model(mnm.Model):
            # atan --> allreduce -> atan ---> mul
            #                   \-> atan --/
            def build(self):
                pass

            @mnm.model.trace
            def forward(self, x):
                a = mnm.atan(x)
                b = mnm.allreduce([a])
                c = mnm.atan(b)
                d = mnm.atan(b)
                e = mnm.multiply(c, d)
                return e

        model = Model()
        x, _ = randn(shape)
        mod = model._internal(x).mod

        mod = MNMSequential([AnnotateDistOps()])(mod)

        def expected():
            """
            fn (%x: Tensor[(64, 128), float32]) {
                let %set_stream_comp = mnm.op.set_stream(int64(0), int64(1));
                let %a1 = mnm.op.atan(%x);
                let %a2 = (%a1,);
                let %add_event_comp = mnm.op.add_event(int64(1), int64(1));
                let %set_stream_comm = mnm.op.set_stream(int64(0), int64(5));
                let %wait_for_comp = mnm.op.wait_event(int64(1), int64(5));
                let %a3 = mnm.op._allreduce(%a2, str"sum");
                let %add_event_comm = mnm.op.add_event(int64(2), int64(5));
                let %set_stream_comp1 = mnm.op.set_stream(int64(0), int64(1));
                let %wait_for_comm = mnm.op.wait_event(int64(2), int64(1));
                let %a4 = mnm.op.atan(%a3);
                let %a5 = mnm.op.atan(%a3);
                let %a6 = mnm.op.multiply(%a4, %a5);
                %a6
            }
            """
            builder = ANFBuilder()
            x = extended_var("x", shape=shape)
            builder.set_stream(0, comp_stream)

            x_1 = builder.call("atan", [x])
            x_2 = builder.make_tuple([x_1])
            builder.add_event(1, comp_stream)
            builder.set_stream(0, comm_stream)
            builder.wait_event(1, comm_stream)
            x_3 = builder.call("_allreduce", [x_2, mnm.ir.const("sum")])
            builder.add_event(4, comm_stream)
            builder.set_stream(0, comp_stream)
            builder.wait_event(4, comp_stream)
            x_4_a = builder.call("atan", [x_3])
            x_4_b = builder.call("atan", [x_3])
            x_5 = builder.call("multiply", [x_4_a, x_4_b])

            return tvm.relay.Function([x], builder.ret(x_5))

    assert tvm.ir.structural_equal(mod['main'], expected())
    dctx.enable_data_parallel = False


if __name__ == "__main__":
    pytest.main([__file__])
