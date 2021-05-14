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
    default_vec = mnm.ir.const([])
    null = mnm.ir.const(None)

    def expected1():
        x = mnm.ir.var("x", shape=(10, 20))
        dy = mnm.ir.var("dy", shape=(10, 15))
        w = mnm.ir.var("dy", shape=(20, 15))
        a1 = mnm.ir.var("a1")
        a2 = mnm.ir.var("a2")
        a3 = mnm.ir.var("a3")
        a5 = mnm.ir.var("a5")
        let5 = relay.Let(a5, relay.Call(subtract_op, [w, a2, null, null]), a5)
        let3 = relay.Let(a3, relay.Tuple([a2]), let5)
        let2 = relay.Let(a2, relay.Call(matmul_op, [a1, dy]), let3)
        let1 = relay.Let(a1, relay.Call(transpose_op, [x, default_vec]), let2)
        return relay.Function([x, dy, w], let1)

    def expected2():
        x = mnm.ir.var("x", shape=(10, 20))
        dy = mnm.ir.var("dy", shape=(10, 15))
        w = mnm.ir.var("dy", shape=(20, 15))
        a1 = mnm.ir.var("a1")
        a2 = mnm.ir.var("a2")
        a5 = mnm.ir.var("a5")
        let5 = relay.Let(a5, relay.Call(subtract_op, [w, a2, null, null]), a5)
        let2 = relay.Let(a2, relay.Call(matmul_op, [a1, dy]), let5)
        let1 = relay.Let(a1, relay.Call(transpose_op, [x, default_vec]), let2)
        return relay.Function([x, dy, w], let1)

    def expected3():
        x = mnm.ir.var("p0", shape=(20, 15))
        y = mnm.ir.var("p1", shape=(20, 10))
        z = mnm.ir.var("p2", shape=(10, 15))
        o = relay.Call(matmul_op, [y, z])
        o = relay.Call(subtract_op, [x, o, null, null])
        f = relay.Function([x, y, z], o)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = mnm.ir.var("x", shape=(10, 20))
        dy = mnm.ir.var("dy", shape=(10, 15))
        w = mnm.ir.var("dy", shape=(20, 15))
        out = relay.Call(transpose_op, [x, default_vec])
        out = relay.Call(f, [w, out, dy])
        return relay.Function([x, dy, w], out)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    m_dy, _ = randn((10, 15), device="cpu")
    mod = model._internal(m_x, m_dy).mod



    mod = run_infer_type(mod)
    mod = run_infer_type(mnm._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected1())
    assert tvm.ir.structural_equal(mod['main'], func_expected)

    mod = run_infer_type(mnm._ffi.pass_.DeadCodeElimination()(mod))
    func_expected = run_infer_type(expected2())
    assert tvm.ir.structural_equal(mod['main'], func_expected)

    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = mnm._ffi.pass_.FuseOps()(mod)
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
        x = mnm.ir.var("x", shape=shape)
        y = mnm.ir.var("y", shape=shape)
        a1 = mnm.ir.var("a1")
        a2 = mnm.ir.var("a2")
        a3 = mnm.ir.var("a3")
        let3 = relay.Let(a3, relay.Tuple([a1, a2]), x)
        let2 = relay.Let(a2, relay.Tuple([y,]), let3)
        let1 = relay.Let(a1, relay.Tuple([x,]), let2)
        return relay.Function([x, y], let1)

    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    model = Model()
    mod = model._internal(m_x, m_y).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(mnm._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod['main'], func_expected)


def test_tuple_sequence():
    shape = (10, 20)
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            a = Symbol.make_tuple([x,])
            a = Symbol.make_tuple([a[0],])
            a = mnm.add(a[0], y)
            return a

    def expected():
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        null = mnm.ir.const(None)
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        let3 = relay.Let(a3, relay.Call(add_op, [x, y, null, null]), a3)
        let2 = relay.Let(a2, relay.Tuple([x,]), let3)
        let1 = relay.Let(a1, relay.Tuple([x,]), let2)
        return relay.Function([x, y], let1)

    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    model = Model()
    mod = model._internal(m_x, m_y).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(mnm._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod['main'], func_expected)

if __name__ == "__main__":
    pytest.main([__file__])
