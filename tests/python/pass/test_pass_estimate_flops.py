# pylint: disable=protected-access
import pytest
import mnm
import tvm
from tvm import relay

from mnm._ffi.pass_ import EstimateFLOPS
from mnm.ir import ScopeBuilder
from mnm.testing import run_infer_type

def verify_flops(mod, expected_map):
    with tvm.target.Target("llvm"):
        ret = EstimateFLOPS(run_infer_type(mod))
        ret = {k.name_hint: v.value for k, v in ret.items()}

    for var_name, expected_flops in expected_map.items():
        assert var_name in ret, "Missing %s" % var_name
        assert abs(expected_flops - ret[var_name]) <= 1, "%s FLOPS mismatch" % var_name

def test_conv2d():
    shape = (16, 16, 64, 64)

    def get_mod():
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        conv2d_call = lambda x, w: relay.Call(conv2d_op,
                                              [x, w, mnm.ir.const([1]), mnm.ir.const([1]),
                                               mnm.ir.const([1]), mnm.ir.const(1),
                                               mnm.ir.const("NCHW"), mnm.ir.const("OIHW"),
                                               mnm.ir.const("NCHW")])

        data = mnm.ir.var("x", shape=shape)
        weight = mnm.ir.var("w", shape=(16, 16, 3, 3))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", conv2d_call(data, weight))
        sb.ret(a_1)
        func = relay.Function([data, weight], sb.get())
        return tvm.IRModule.from_expr(func)

    # 2 * (N * CI * CO * H * W * kh * kw)
    verify_flops(get_mod(), {"a1": 2 * 16 * 16 * 16 * 64 * 64 * 3 * 3})

def test_unary():
    shape = (10, 5)

    def get_mod():
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        data = mnm.ir.var("x", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [data]))
        a_2 = sb.let("a2", relay.Call(relu_op, [a_1]))
        sb.ret(a_2)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    verify_flops(get_mod(), {"a1": 10 * 5, "a2": 10 * 5})


def test_fusion():
    shape = (10, 5)

    def get_mod():
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        data = mnm.ir.var("x", shape=shape)

        p_0 = mnm.ir.var("p0", shape=shape)
        out = relay.Call(relu_op, [p_0])
        out = relay.Call(relu_op, [out])
        closure = relay.Function([p_0], out)
        closure = closure.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(closure, [data]))
        sb.ret(a_1)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    verify_flops(get_mod(), {"a1": 10 * 5 * 2})

def test_multi_func():
    shape = (10, 5)

    def get_mod():
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")

        sb = ScopeBuilder()
        data = mnm.ir.var("x", shape=shape)
        a_1 = sb.let("a1", relay.Call(relu_op, [data]))
        a_2 = sb.let("a2", relay.Call(relu_op, [a_1]))
        sb.ret(a_2)

        mod = tvm.IRModule()
        func_1 = relay.GlobalVar("func_1")
        mod[func_1] = relay.Function([data], sb.get())

        sb = ScopeBuilder()
        data = mnm.ir.var("x", shape=shape)
        b_1 = sb.let("b1", relay.Call(func_1, [data]))
        sb.ret(b_1)
        mod[relay.GlobalVar("main")] = relay.Function([data], sb.get())
        return mod

    verify_flops(get_mod(), {"b1": 10 * 5 * 2})


if __name__ == "__main__":
    pytest.main([__file__])
