import pytest
import torch
import mnm
from mnm._ffi.pass_ import AutoDiff
from mnm.testing import check_type, run_infer_type, randn, randn_torch
from tvm.relay import TensorType, FuncType, TupleType


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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d(dtype, xshape, wshape, stride, dilation, padding): # pylint: disable=too-many-arguments
    # N.B.: NCHW + OIHW
    import torch.nn.functional as F

    class Conv2D(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w):  # pylint: disable=no-self-use
            return mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    class Conv2DGrad(mnm.Model):
        def build(self, grad_mode):
            self._grad_mode = grad_mode

        @mnm.model.trace
        def forward(self, x_or_w, y, dy): # pylint: disable=inconsistent-return-statements
            if self._grad_mode == "dx":
                return mnm.conv2d_dx(x_or_w, y, dy, shape=xshape,
                                     stride=stride, padding=padding, dilation=dilation, groups=1)
            if self._grad_mode == "dw":
                return mnm.conv2d_dw(x_or_w, y, dy, shape=wshape,
                                     stride=stride, padding=padding, dilation=dilation, groups=1)

    model = Conv2D()
    # forward
    m_x, t_x = randn_torch(xshape, std=0.001, dtype=dtype)
    m_w, t_w = randn_torch(wshape, std=0.01, dtype=dtype)
    m_y = model(m_x, m_w)
    m_func = model.get_relay_func(m_x, m_w)
    m_func = run_infer_type(m_func)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    x_ty = TensorType(xshape, dtype=dtype)
    w_ty = TensorType(wshape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, w_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    # TODO(@XIAO-XIA): using AutoDiff to check backward after impl the type func of shape
    dx_modle = Conv2DGrad("dx")
    dw_modle = Conv2DGrad("dw")
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype)
    dy_ty = TensorType(t_dy.shape, dtype=dtype)
    dx_func = dx_modle.get_relay_func(m_w, m_y, m_dy)
    dw_func = dw_modle.get_relay_func(m_x, m_y, m_dy)
    dx_func = run_infer_type(dx_func)
    dw_func = run_infer_type(dw_func)
    dx_checked_type = FuncType([w_ty, y_ty, dy_ty], x_ty)
    dw_checked_type = FuncType([x_ty, y_ty, dy_ty], w_ty)
    check_type(dx_func, dx_checked_type)
    check_type(dw_func, dw_checked_type)


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
