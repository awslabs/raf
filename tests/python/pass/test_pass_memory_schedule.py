# pylint: disable=protected-access,too-many-locals
import pytest
import mnm
import tvm
from tvm import relay
from mnm._ffi.pass_ import InferType, MemorySchedule
from mnm.ir import ScopeBuilder


def check_ir(mod, expected):
    with mnm.ir.PassContext(config={"mnm.memory_schedule": True}):
        mod = MemorySchedule()(mod)

    assert tvm.ir.structural_equal(mod["main"], expected["main"]), "IR mismatch"


def test_simple():
    shape = (1024, 1024)
    shape2 = (2048, 1024)

    def get_mod_n_expected():
        null = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        sum_op = mnm._ffi.op.GetOp("mnm.op.sum")
        concat_op = mnm._ffi.op.GetOp("mnm.op.concatenate")

        # module to work on.
        sb = ScopeBuilder()
        param0 = mnm.ir.var("param0", shape=shape)
        param1 = mnm.ir.var("param1", shape=shape)
        param2 = mnm.ir.var("param2", shape=shape2)
        a_3 = sb.let("a3", relay.Call(relu_op, [param0]))
        a_9 = sb.let("a9", relay.Call(relu_op, [a_3]))
        a_1 = sb.let("a1", relay.Call(relu_op, [param1]))
        a_4 = sb.let("a4", relay.Tuple([a_3, a_1]))
        a_5 = sb.let("a5", relay.Call(concat_op, [a_4]))
        a_6 = sb.let("a6", relay.Call(add_op, [a_5, param2, null, null]))
        a_7 = sb.let("a7", relay.Call(sum_op, [a_6, mnm.ir.const(0)]))
        a_2 = sb.let("a2", relay.Call(sum_op, [a_1, mnm.ir.const(0)]))
        a_8 = sb.let("a8", relay.Call(add_op, [a_2, a_7, null, null]))
        a_10 = sb.let("a10", relay.Call(add_op, [a_8, a_9]))
        sb.ret(a_10)
        func = relay.Function([param0, param1, param2], sb.get())
        mod = tvm.IRModule.from_expr(func)

        # expected schedule.
        sb = ScopeBuilder()
        param0 = mnm.ir.var("param0", shape=shape)
        param1 = mnm.ir.var("param1", shape=shape)
        param2 = mnm.ir.var("param2", shape=shape2)
        a_1 = sb.let("a1", relay.Call(relu_op, [param1]))
        a_2 = sb.let("a2", relay.Call(sum_op, [a_1, mnm.ir.const(0)]))
        a_3 = sb.let("a3", relay.Call(relu_op, [param0]))
        a_4 = sb.let("a4", relay.Tuple([a_3, a_1]))
        a_5 = sb.let("a5", relay.Call(concat_op, [a_4]))
        a_6 = sb.let("a6", relay.Call(add_op, [a_5, param2, null, null]))
        a_7 = sb.let("a7", relay.Call(sum_op, [a_6, mnm.ir.const(0)]))
        a_8 = sb.let("a8", relay.Call(add_op, [a_2, a_7, null, null]))
        a_9 = sb.let("a9", relay.Call(relu_op, [a_3]))
        a_10 = sb.let("a10", relay.Call(add_op, [a_8, a_9]))
        sb.ret(a_10)
        func = relay.Function([param0, param1, param2], sb.get())
        expected = tvm.IRModule.from_expr(func)
        return InferType()(mod), InferType()(expected)

    check_ir(*get_mod_n_expected())


def test_no_let():
    def get_mod_n_expected():
        sb = ScopeBuilder()
        sb.ret(mnm.ir.const(1))
        func = relay.Function([], sb.get())
        mod = tvm.IRModule.from_expr(func)
        return InferType()(mod), InferType()(mod)

    check_ir(*get_mod_n_expected())


if __name__ == "__main__":
    pytest.main([__file__])
