# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-statements
import pytest
import mnm
from mnm._core.ndarray import Symbol
from mnm.testing import randn, run_infer_type
import tvm
from tvm import relay


def test_inline():
    class Model(mnm.Model):
        def build(self):
            self.w, _ = randn((20, 15), ctx="cpu")

        @mnm.model.trace
        def forward(self, x, dy):
            xt = mnm.transpose(x)
            dw = mnm.matmul(xt, dy)
            tup = Symbol.make_tuple([dw._Symbol__handle,])
            it = tup[0]
            y = mnm.subtract(self.w, it)
            return y

    transpose_op = mnm._ffi.op.GetOp("mnm.op.transpose")
    matmul_op = mnm._ffi.op.GetOp("mnm.op.matmul")
    subtract_op = mnm._ffi.op.GetOp("mnm.op.subtract")
    default_int = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))
    default_vec = mnm._ffi.ir._make.Constant(mnm._core.value.TupleValue([]))

    def expected1():
        x = relay.var("x", shape=(10, 20))
        dy = relay.var("dy", shape=(10, 15))
        w = relay.var("dy", shape=(20, 15))
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        a5 = relay.var("a5")
        let5 = relay.Let(a5, relay.Call(subtract_op, [w, a2, default_int, default_int]), a5)
        let3 = relay.Let(a3, relay.Tuple([a2]), let5)
        let2 = relay.Let(a2, relay.Call(matmul_op, [a1, dy]), let3)
        let1 = relay.Let(a1, relay.Call(transpose_op, [x, default_vec]), let2)
        return relay.Function([x, dy, w], let1)

    def expected2():
        x = relay.var("x", shape=(10, 20))
        dy = relay.var("dy", shape=(10, 15))
        w = relay.var("dy", shape=(20, 15))
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a5 = relay.var("a5")
        let5 = relay.Let(a5, relay.Call(subtract_op, [w, a2, default_int, default_int]), a5)
        let2 = relay.Let(a2, relay.Call(matmul_op, [a1, dy]), let5)
        let1 = relay.Let(a1, relay.Call(transpose_op, [x, default_vec]), let2)
        return relay.Function([x, dy, w], let1)

    def expected3():
        x = relay.var("p0", shape=(10, 20))
        y = relay.Call(transpose_op, [x, default_vec])
        f1 = relay.Function([x], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("p01", shape=(20, 10))
        y = relay.var("p1", shape=(10, 15))
        z = relay.var("p2", shape=(20, 15))
        o = relay.Call(matmul_op, [x, y])
        o = relay.Call(subtract_op, [z, o, default_int, default_int])
        f2 = relay.Function([x, y, z], o)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=(10, 20))
        dy = relay.var("dy", shape=(10, 15))
        w = relay.var("dy", shape=(20, 15))
        a1 = relay.var("a1")
        a5 = relay.var("a5")
        let5 = relay.Let(a5, relay.Call(f2, [a1, dy, w]), a5)
        let1 = relay.Let(a1, relay.Call(f1, [x]), let5)
        return relay.Function([x, dy, w], let1)

    model = Model()
    m_x, _ = randn((10, 20), ctx="cpu")
    m_dy, _ = randn((10, 15), ctx="cpu")
    func = model._internal(m_x, m_dy).func
    func = run_infer_type(func)
    func = run_infer_type(mnm._ffi.pass_.InlineLet(func))
    func_expected = run_infer_type(expected1())
    assert tvm.ir.structural_equal(func, func_expected)

    func = run_infer_type(mnm._ffi.pass_.DeadCodeElimination(func))
    func_expected = run_infer_type(expected2())
    assert tvm.ir.structural_equal(func, func_expected)

    func = run_infer_type(mnm._ffi.pass_.FuseOps(func, 3))
    func_expected = run_infer_type(expected3())
    assert tvm.ir.structural_equal(func, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
