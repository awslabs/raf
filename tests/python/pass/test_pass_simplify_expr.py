# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, no-self-use, too-many-locals
import pytest
import raf
from raf._core.ir_ext import extended_var
from raf._core.ndarray import array
from raf._ffi.pass_ import SimplifyExpr, ToGraphNormalForm, ToBasicBlockNormalForm, InferType
from raf.ir import RAFSequential, ScopeBuilder
from raf.testing import randn

import tvm
from tvm import relay


def simplify(mod, device):
    with raf.Device(device):
        seq = RAFSequential([ToGraphNormalForm(), ToBasicBlockNormalForm(), SimplifyExpr()])
        return seq(mod)


@pytest.mark.parametrize("op", ["zeros_like", "ones_like"])
def test_unary_like(op):
    device = "cpu"
    shape = (10, 5)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return getattr(raf._op.sym, op)(x)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")

    mod = model._internal(m_x).mod
    mod = simplify(mod, device)
    text = raf.ir.AsText(mod["main"])
    assert "like" not in text, text


@pytest.mark.parametrize(
    "params", [("cast_like", (10, 1), "float16"), ("broadcast_to_like", (10, 10), "float32")]
)
def test_binary_like(params):
    op, shape_like, dtype_like = params
    device = "cpu"
    shape = (10, 1)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            return getattr(raf._op.sym, op)(x, y)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    m_y, _ = randn(shape_like, device=device, dtype=dtype_like)

    mod = model._internal(m_x, m_y).mod
    mod = simplify(mod, device)
    text = raf.ir.AsText(mod["main"])
    assert "like" not in text, text


def test_cast():
    device = "cpu"
    shape = (10, 5)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.cast(x, "float16")
            y = raf.cast(y, "float32")
            y = raf.cast(y, "float16")
            y = raf.cast(y, "float32")
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod, device)
    text = raf.ir.AsText(mod["main"])
    assert "raf.op.cast" not in text, text


@pytest.mark.parametrize("t_endpoints", ["float32", "int32", "uint64", "bool"])
@pytest.mark.parametrize("t_middle", ["float32", "int32", "uint64", "bool"])
def test_cast_across_type(t_endpoints, t_middle):
    device = "cpu"
    shape = (10, 5)

    cast_level_map = {"bool": 4, "uint64": 3, "int32": 2, "float32": 1}
    should_simplify = cast_level_map[t_endpoints] >= cast_level_map[t_middle]

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.cast(x, t_middle)
            x = raf.cast(x, t_endpoints)
            return x

    model = Model()
    m_x, _ = randn(shape, device=device, dtype=t_endpoints)
    mod = model._internal(m_x).mod
    mod = simplify(mod, device)
    text = raf.ir.AsText(mod["main"])
    if should_simplify:
        assert "raf.op.cast" not in text, text
    else:
        assert "raf.op.cast" in text, text


def test_reshape():
    device = "cpu"
    shape = (10, 5)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.reshape(x, (shape[0] * shape[1],))
            y = raf.reshape(y, (shape[1], shape[0]))
            y = raf.reshape(y, shape)
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod, device)
    text = raf.ir.AsText(mod["main"])
    assert "raf.op.reshape" not in text, text


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("act", [False, True])
@pytest.mark.parametrize("shape_compatible", [False, True])
def test_matmul_reshape_bias(ndim, act, shape_compatible):
    device = "cpu"
    xshape = (10, 10) if ndim == 2 else (1, 10, 10)
    b2shape = (2, 5, 10)
    bshape = (10,) if shape_compatible else (2, 5, 10)
    matmul_op = getattr(raf._op.sym, "matmul" if ndim == 2 else "batch_matmul")

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w, b):
            y = matmul_op(x, w)
            y = raf.reshape(y, b2shape)
            y = raf.add(y, b)
            y = raf.gelu(y) if act else y
            return y

    model = Model()
    m_x, _ = randn(xshape, device=device, dtype="float32")
    m_w, _ = randn(xshape, device=device, dtype="float32")
    m_b, _ = randn(bshape, device=device, dtype="float32")
    mod = model._internal(m_x, m_w, m_b).mod
    mod = simplify(mod, device)

    def expected():
        matmul_op = raf._ffi.op.GetOp("raf.op.%s" % ("matmul" if ndim == 2 else "batch_matmul"))
        add_op = raf._ffi.op.GetOp("raf.op.add")
        gelu_op = raf._ffi.op.GetOp("raf.op.gelu")
        reshape_op = raf._ffi.op.GetOp("raf.op.reshape")
        null = raf.ir.const(None)

        x = extended_var("x", shape=xshape, dtype="float32")
        w = extended_var("w", shape=xshape, dtype="float32")
        b = extended_var("b", shape=bshape, dtype="float32")
        y = relay.Call(matmul_op, [x, w])
        if not shape_compatible:
            y = relay.Call(reshape_op, [y, raf.ir.const(b2shape), raf.ir.const(False)])
        y = relay.Call(add_op, [y, b, null, null])
        if act:
            y = relay.Call(gelu_op, [y])
        if shape_compatible:
            y = relay.Call(reshape_op, [y, raf.ir.const(b2shape), raf.ir.const(False)])
        mod = tvm.IRModule.from_expr(relay.Function([x, w, b], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), raf.ir.AsText(mod["main"])


def test_multiply():
    device = "cpu"
    shape = (10, 5)
    mul_op = raf._ffi.op.GetOp("raf.op.multiply")

    data_x = raf.ir.var("x", shape=shape, dtype="float32")
    const_1 = raf.ir.const(1.0, dtype="float32")
    const_0 = raf.ir.const(0.0, dtype="float32")
    tensor_1 = raf.ir.const(array(1, dtype="float32", device=device))
    tensor_0 = raf.ir.const(array(0, dtype="float32", device=device))

    sb = ScopeBuilder()
    a_1 = sb.let("a1", relay.Call(mul_op, [data_x, const_1]))
    a_2 = sb.let("a2", relay.Call(mul_op, [a_1, tensor_1]))
    a_3 = sb.let("a3", relay.Call(mul_op, [a_2, const_0]))
    a_4 = sb.let("a4", relay.Call(mul_op, [tensor_0, a_3]))
    sb.ret(a_4)
    func = relay.Function([data_x], sb.get())
    mod = tvm.IRModule.from_expr(func)
    mod = simplify(mod, device)

    def expected():
        zeros_op = raf._ffi.op.GetOp("raf.op.zeros")
        x = extended_var("x", shape=shape, dtype="float32")
        y = relay.Call(zeros_op, [raf.ir.const(shape), raf.ir.const("float32")])
        mod = tvm.IRModule.from_expr(relay.Function([x], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), raf.ir.AsText(mod["main"])


def test_add_sub():
    device = "cpu"
    shape = (10, 5)
    add_op = raf._ffi.op.GetOp("raf.op.add")
    sub_op = raf._ffi.op.GetOp("raf.op.subtract")

    data_x = raf.ir.var("x", shape=shape, dtype="float32")
    const_0 = raf.ir.const(0.0, dtype="float32")
    tensor_0 = raf.ir.const(array(0, dtype="float32", device=device))
    null = raf.ir.const(None)

    sb = ScopeBuilder()
    a_1 = sb.let("a1", relay.Call(sub_op, [const_0, data_x, null, null]))
    a_2 = sb.let("a2", relay.Call(add_op, [a_1, const_0, null, null]))
    a_3 = sb.let("a3", relay.Call(add_op, [a_2, tensor_0, null, null]))
    a_4 = sb.let("a4", relay.Call(add_op, [a_3, const_0, data_x, null]))
    sb.ret(a_4)
    func = relay.Function([data_x], sb.get())
    mod = tvm.IRModule.from_expr(func)
    mod = simplify(mod, device)

    def expected():
        x = extended_var("x", shape=shape, dtype="float32")
        y = relay.Call(sub_op, [const_0, x, null, null])
        y = relay.Call(add_op, [y, const_0, x, null])
        mod = tvm.IRModule.from_expr(relay.Function([x], y))
        return InferType()(mod)["main"]

    assert tvm.ir.structural_equal(mod["main"], expected()), raf.ir.AsText(mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
