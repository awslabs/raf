import numpy as np
import pytest
import torch
import mnm
from mnm._ffi.pass_ import InferType, AutoDiff
from tvm import relay
from tvm.relay import TensorType, FuncType, TupleType


def check_type(expr, typ):
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randn_torch(shape, *, ctx="cpu", dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    t_x = torch.tensor(n_x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def run_infer_type(func):
    # pylint: disable=protected-access
    mod = mnm._ffi.ir._make.Module({relay.GlobalVar("main"): func})
    mod = InferType(mod)
    return mod['main']


# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [
    [3],
    [3, 2],
    [3, 2, 5],
    [3, 2, 5, 8],
    [3, 2, 5, 8, 4],
    [3, 2, 5, 8, 4, 7],
])
@pytest.mark.parametrize("axis", range(-8, 8))
@pytest.mark.parametrize(
    "funcs",
    [
        [mnm._op.sym.softmax, torch.softmax],
        [mnm._op.sym.log_softmax, torch.log_softmax],
    ])
def test_unary_with_axis(dtype, shape, axis, funcs):
    mnm_fwd, torch_fwd = funcs

    class Softmax(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            return mnm_fwd(x, axis=axis)

    model = Softmax()
    # forward
    m_x, t_x = randn_torch(shape, dtype=dtype)
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_func = model.get_relay_func(m_x)
            m_func = run_infer_type(m_func)
        return
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    t_y = torch_fwd(t_x, dim=axis)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_func, checked_type)
    # backward
    _, t_dy = randn_torch(shape, dtype=dtype)
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    t_y.backward(t_dy)
    dy_ty = TensorType(t_dy.shape, dtype=dtype)
    dx_ty = TensorType(t_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    checked_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, checked_type)


# pylint: disable=import-outside-toplevel, attribute-defined-outside-init
@pytest.mark.parametrize("shape", [
    (5, 4, 6, 9),
    (6, 5, 7, 10),
    (12, 32, 6, 8),
    (3, 7, 9)
])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("eps", [1e-05, 2e-05])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm(shape, axis, eps, dtype):
    import mxnet as mx

    class LayerNorm(mnm.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @mnm.model.trace
        def forward(self, x):
            return mnm.layer_norm(x, axis=self._axis, eps=self._eps)

    model = LayerNorm(axis, eps)
    mx_model = mx.gluon.nn.LayerNorm(axis=axis, epsilon=eps, center=False, scale=False)
    mx_model.initialize(ctx=mx.cpu(0))
    # forward
    m_x, n_x = randn(shape, dtype=dtype)
    mx_x = mx.nd.array(n_x)
    mx_x.attach_grad()
    m_y = model(m_x)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    _, n_dy = randn(m_y.shape, dtype=dtype)
    mx_dy = mx.nd.array(n_dy)
    with mx.autograd.record():
        mx_y = mx_model(mx_x)
        mx_y.backward(mx_dy)
    x_ty = TensorType(mx_x.shape, dtype=dtype)
    y_ty = TensorType(mx_y.shape, dtype=dtype)
    dy_ty = TensorType(mx_dy.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_func, checked_type)
    # check backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dx_ty = TensorType(mx_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    checked_type = FuncType([y_ty], TupleType([x_ty, bwd_ty]))
    check_type(m_func, checked_type)


# pylint: disable=no-self-use, too-many-arguments
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("data_shape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize(
    "funcs",
    [
        [mnm._op.sym.max_pool2d, torch.nn.functional.max_pool2d],
        [mnm._op.sym.avg_pool2d, torch.nn.functional.avg_pool2d],
    ])
def test_pool2d(dtype, data_shape, kernel, stride, padding, funcs):

    mnm_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class Pool2D(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            return mnm_fwd(x, kernel=kernel, stride=stride, padding=padding)

    model = Pool2D()
    # forward
    m_x, t_x = randn_torch(data_shape, dtype=dtype)
    m_y = model(m_x)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_func, checked_type)
    # backward
    _, t_dy = randn_torch(m_y.shape, dtype=dtype)
    t_y.backward(t_dy)
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dy_ty = TensorType(t_dy.shape, dtype=dtype)
    dx_ty = TensorType(t_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    checked_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, checked_type)


if __name__ == "__main__":
    pytest.main([__file__])
