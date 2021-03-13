# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-statements
# pylint: disable=no-self-use
import pytest
import mnm
from mnm._core.ndarray import Symbol
from mnm.testing import randn, run_infer_type
import tvm
from tvm import relay


def test_inline():
    class Model(mnm.Model):
        def build(self):
            self.w, _ = randn((20, 15), device="cpu")

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
        let1 = relay.Let(a1, relay.Call(transpose_op, [x, default_vec]), let5)
        return relay.Function([x, dy, w], let1)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    m_dy, _ = randn((10, 15), device="cpu")
    mod = model._internal(m_x, m_dy).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(mnm._ffi.pass_.InlineLet(mod))
    func_expected = run_infer_type(expected1())
    assert tvm.ir.structural_equal(mod['main'], func_expected)

    mod = run_infer_type(mnm._ffi.pass_.DeadCodeElimination(mod))
    func_expected = run_infer_type(expected2())
    assert tvm.ir.structural_equal(mod['main'], func_expected)

    mod = run_infer_type(mnm._ffi.pass_.FuseOps(mod, 3))
    func_expected = run_infer_type(expected3())
    assert tvm.ir.structural_equal(mod['main'], func_expected)


def test_nested_tuple():
    shape = (10, 20)
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            tup1 = Symbol.make_tuple([x,])
            tup2 = Symbol.make_tuple([y,])
            tup = Symbol.make_tuple([tup1, tup2])
            x1 = tup[0]
            y1 = x1[0]
            return y1

    def expected():
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        let3 = relay.Let(a3, relay.Tuple([a1, a2]), x)
        let2 = relay.Let(a2, relay.Tuple([y,]), let3)
        let1 = relay.Let(a1, relay.Tuple([x,]), let2)
        return relay.Function([x, y], let1)

    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    model = Model()
    mod = model._internal(m_x, m_y).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(mnm._ffi.pass_.InlineLet(mod))
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod['main'], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
