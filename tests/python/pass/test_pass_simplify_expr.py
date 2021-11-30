# pylint: disable=protected-access, no-self-use, too-many-locals
import pytest
import mnm
from mnm._core.ir_ext import extended_var
from mnm._core.ndarray import array
from mnm._ffi.pass_ import SimplifyExpr, ToGraphNormalForm, ToBasicBlockNormalForm, InferType
from mnm.ir import MNMSequential, ScopeBuilder
from mnm.testing import randn

import tvm
from tvm import relay


def simplify(mod):
    seq = MNMSequential([ToGraphNormalForm(), ToBasicBlockNormalForm(), SimplifyExpr()])
    return seq(mod)


@pytest.mark.parametrize("op", ["zeros_like", "ones_like"])
def test_unary_like(op):
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return getattr(mnm._op.sym, op)(x)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")

    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "like" not in text, text


@pytest.mark.parametrize(
    "params", [("cast_like", (10, 1), "float16"), ("broadcast_to_like", (10, 10), "float32")]
)
def test_binary_like(params):
    op, shape_like, dtype_like = params
    device = "cpu"
    shape = (10, 1)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return getattr(mnm._op.sym, op)(x, y)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    m_y, _ = randn(shape_like, device=device, dtype=dtype_like)

    mod = model._internal(m_x, m_y).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "like" not in text, text


def test_cast():
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.cast(x, "float16")
            y = mnm.cast(y, "float32")
            y = mnm.cast(y, "float16")
            y = mnm.cast(y, "float32")
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "mnm.op.cast" not in text, text


def test_reshape():
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.reshape(x, (shape[0] * shape[1],))
            y = mnm.reshape(y, (shape[1], shape[0]))
            y = mnm.reshape(y, shape)
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "mnm.op.reshape" not in text, text


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("act", [False, True])
def test_matmul_reshape_bias(ndim, act):
    device = "cpu"
    xshape = (10, 10) if ndim == 2 else (1, 10, 10)
    bshape = (10,)
    matmul_op = getattr(mnm._op.sym, "matmul" if ndim == 2 else "batch_matmul")

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w, b):
            y = matmul_op(x, w)
            y = mnm.reshape(y, (2, 5, 10))
            y = mnm.add(y, b)
            y = mnm.gelu(y) if act else y
            return y

    model = Model()
    m_x, _ = randn(xshape, device=device, dtype="float32")
    m_w, _ = randn(xshape, device=device, dtype="float32")
    m_b, _ = randn(bshape, device=device, dtype="float32")
    mod = model._internal(m_x, m_w, m_b).mod
    mod = simplify(mod)

    def expected():
        matmul_op = mnm._ffi.op.GetOp("mnm.op.%s" % ("matmul" if ndim == 2 else "batch_matmul"))
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        gelu_op = mnm._ffi.op.GetOp("mnm.op.gelu")
        reshape_op = mnm._ffi.op.GetOp("mnm.op.reshape")
        null = mnm.ir.const(None)

        x = extended_var("x", shape=xshape, dtype="float32")
        w = extended_var("w", shape=xshape, dtype="float32")
        b = extended_var("b", shape=bshape, dtype="float32")
        y = relay.Call(matmul_op, [x, w])
        y = relay.Call(add_op, [y, b, null, null])
        if act:
            y = relay.Call(gelu_op, [y])
        y = relay.Call(reshape_op, [y, mnm.ir.const((2, 5, 10)), mnm.ir.const(False)])
        mod = tvm.IRModule.from_expr(relay.Function([x, w, b], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), mnm.ir.AsText(mod["main"])


def test_multiply():
    device = "cpu"
    shape = (10, 5)
    mul_op = mnm._ffi.op.GetOp("mnm.op.multiply")

    data_x = mnm.ir.var("x", shape=shape, dtype="float32")
    const_1 = mnm.ir.const(1.0, dtype="float32")
    const_0 = mnm.ir.const(0.0, dtype="float32")
    tensor_1 = mnm.ir.const(array(1, dtype="float32", device=device))
    tensor_0 = mnm.ir.const(array(0, dtype="float32", device=device))

    sb = ScopeBuilder()
    a_1 = sb.let("a1", relay.Call(mul_op, [data_x, const_1]))
    a_2 = sb.let("a2", relay.Call(mul_op, [a_1, tensor_1]))
    a_3 = sb.let("a3", relay.Call(mul_op, [a_2, const_0]))
    a_4 = sb.let("a4", relay.Call(mul_op, [tensor_0, a_3]))
    sb.ret(a_4)
    func = relay.Function([data_x], sb.get())
    mod = tvm.IRModule.from_expr(func)
    mod = simplify(mod)

    def expected():
        zeros_op = mnm._ffi.op.GetOp("mnm.op.zeros")
        x = extended_var("x", shape=shape, dtype="float32")
        y = relay.Call(zeros_op, [mnm.ir.const(shape), mnm.ir.const("float32")])
        mod = tvm.IRModule.from_expr(relay.Function([x], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), mnm.ir.AsText(mod["main"])


def test_add_sub():
    device = "cpu"
    shape = (10, 5)
    add_op = mnm._ffi.op.GetOp("mnm.op.add")
    sub_op = mnm._ffi.op.GetOp("mnm.op.subtract")

    data_x = mnm.ir.var("x", shape=shape, dtype="float32")
    const_0 = mnm.ir.const(0.0, dtype="float32")
    tensor_0 = mnm.ir.const(array(0, dtype="float32", device=device))
    null = mnm.ir.const(None)

    sb = ScopeBuilder()
    a_1 = sb.let("a1", relay.Call(sub_op, [const_0, data_x, null, null]))
    a_2 = sb.let("a2", relay.Call(add_op, [a_1, const_0, null, null]))
    a_3 = sb.let("a3", relay.Call(add_op, [a_2, tensor_0, null, null]))
    a_4 = sb.let("a4", relay.Call(add_op, [a_3, const_0, data_x, null]))
    sb.ret(a_4)
    func = relay.Function([data_x], sb.get())
    mod = tvm.IRModule.from_expr(func)
    mod = simplify(mod)

    def expected():
        x = extended_var("x", shape=shape, dtype="float32")
        y = relay.Call(sub_op, [const_0, x, null, null])
        y = relay.Call(add_op, [y, const_0, x, null])
        mod = tvm.IRModule.from_expr(relay.Function([x], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), mnm.ir.AsText(mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
