# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import raf
from raf._lib import _TVMError
from raf._ffi.pass_ import AutoDiff, InferType
from raf.testing import check_type, run_infer_type, randn, randn_torch
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8],
    ],
)
@pytest.mark.parametrize("axis", range(-5, 5))
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.softmax, torch.softmax],
        [raf._op.sym.log_softmax, torch.log_softmax],
    ],
)
def test_unary_with_axis(dtype, shape, axis, funcs):
    raf_fwd, torch_fwd = funcs

    class Softmax(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis)

    model = Softmax()
    # forward
    m_x, t_x = randn_torch(shape, dtype=dtype)
    m_x.requires_grad = True
    t_x.requires_grad = True
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(_TVMError):
            m_func = model._internal(m_x).mod["main"]
            m_func = run_infer_type(m_func)
        return
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    t_y = torch_fwd(t_x, dim=axis)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], checked_type)
    # backward
    _, t_dy = randn_torch(shape, dtype=dtype)
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    t_y.backward(t_dy)
    dy_ty = TensorType(t_dy.shape, dtype=dtype)
    dx_ty = TensorType(t_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    checked_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], checked_type)


# pylint: disable=attribute-defined-outside-init
@pytest.mark.parametrize("shape", [(5, 4, 6, 9), (3, 7, 9)])
@pytest.mark.parametrize("eps", [1e-05, 2e-05])
@pytest.mark.parametrize("dtype", ["float32"])
def test_batch_norm_train_dxwb(shape, eps, dtype):
    class BatchNormTrainDxwb(raf.Model):
        def build(self, eps):
            self._eps = eps

        @raf.model.trace
        def forward(self, dy, x, w, b):
            return raf.batch_norm_train_dxwb(dy, x, w, b, self._eps)

    model = BatchNormTrainDxwb(eps)
    # forward
    m_dy, _ = randn(shape, dtype=dtype)
    m_x, _ = randn(shape, dtype=dtype)
    m_w, _ = randn((shape[1],), dtype=dtype)
    m_b, _ = randn((shape[1],), dtype=dtype)
    m_func = model._internal(m_dy, m_x, m_w, m_b).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=dtype)
    w_ty = TensorType((shape[1],), dtype=dtype)
    expected_type = FuncType([x_ty, x_ty, w_ty, w_ty], TupleType([x_ty, w_ty, w_ty]))
    check_type(m_func, expected_type)


# pylint: disable=import-outside-toplevel, attribute-defined-outside-init
@pytest.mark.parametrize("shape", [(5, 4, 6, 9), (3, 7, 9)])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("eps", [1e-05, 2e-05])
@pytest.mark.parametrize("dtype", ["float32"])
def test_layer_norm(shape, axis, eps, dtype):
    import mxnet as mx

    class LayerNorm(raf.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @raf.model.trace
        def forward(self, x, scale, bias):
            return raf.layer_norm(x, scale, bias, axis=self._axis, eps=self._eps)

    model = LayerNorm(axis, eps)
    mx_model = mx.gluon.nn.LayerNorm(axis=axis, epsilon=eps)
    mx_model.initialize(ctx=mx.cpu(0))
    # forward
    m_x, n_x = randn(shape, dtype=dtype)
    m_scale, _ = randn([shape[axis]], dtype=dtype)
    m_bias, _ = randn([shape[axis]], dtype=dtype)
    m_x.requires_grad = True
    mx_x = mx.nd.array(n_x)
    mx_x.attach_grad()

    m_y = model(m_x, m_scale, m_bias)
    record = model._internal(m_x, m_scale, m_bias)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    _, n_dy = randn(m_y.shape, dtype=dtype)
    mx_dy = mx.nd.array(n_dy)
    with mx.autograd.record():
        mx_y = mx_model(mx_x)
        mx_y.backward(mx_dy)
    x_ty = TensorType(mx_x.shape, dtype=dtype)
    scale_ty = TensorType([mx_x.shape[axis]], dtype=dtype)
    bias_ty = TensorType([mx_x.shape[axis]], dtype=dtype)
    y_ty = TensorType(mx_y.shape, dtype=dtype)
    dy_ty = TensorType(mx_dy.shape, dtype=dtype)
    checked_type = FuncType([x_ty, scale_ty, bias_ty], y_ty)
    check_type(m_mod["main"], checked_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    int_type = TensorType((), "int64")
    dx_ty = TensorType(mx_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], TupleType([dx_ty, int_type, int_type]))
    checked_type = FuncType([x_ty, scale_ty, bias_ty], TupleType([x_ty, bwd_ty]))
    check_type(m_mod["main"], checked_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("is_nhwc", [False, True])
def test_conv2d(
    dtype, xshape, wshape, stride, dilation, padding, is_nhwc
):  # pylint: disable=too-many-arguments
    # N.B.: NCHW + OIHW
    import torch.nn.functional as F

    class Conv2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):  # pylint: disable=no-self-use
            layout, kernel_layout = ("NCHW", "OIHW") if not is_nhwc else ("NHWC", "HWIO")
            if is_nhwc:
                x = raf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
                w = raf.transpose(w, (2, 3, 1, 0))  # OIHW -> HWIO
            ret = raf.conv2d(
                x,
                w,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                layout=layout,
                kernel_layout=kernel_layout,
                out_layout=layout,
            )
            if is_nhwc:
                ret = raf.transpose(ret, (0, 3, 1, 2))  # NHWC -> NCHW
            return ret

    class Conv2DGrad(raf.Model):
        def build(self, grad_mode):
            self._grad_mode = grad_mode

        @raf.model.trace
        def forward(self, x_or_w, y, dy):  # pylint: disable=inconsistent-return-statements
            if self._grad_mode == "dx":
                return raf.conv2d_dx(
                    x_or_w,
                    y,
                    dy,
                    shape=xshape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                )
            if self._grad_mode == "dw":
                return raf.conv2d_dw(
                    x_or_w,
                    y,
                    dy,
                    shape=wshape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                )

    model = Conv2D()
    # forward
    m_x, t_x = randn_torch(xshape, std=0.001, dtype=dtype)
    m_w, t_w = randn_torch(wshape, std=0.01, dtype=dtype)
    m_y = model(m_x, m_w)
    m_func = model._internal(m_x, m_w).mod["main"]
    m_func = run_infer_type(m_func)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    x_ty = TensorType(xshape, dtype=dtype)
    w_ty = TensorType(wshape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, w_ty], y_ty)
    check_type(m_func, expected_type)
    if not is_nhwc:
        # NHWC layout is not supported in backward yet.
        # TODO(@XIAO-XIA): using AutoDiff to check backward after impl the type func of shape
        dx_modle = Conv2DGrad("dx")
        dw_modle = Conv2DGrad("dw")
        m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype)
        dy_ty = TensorType(t_dy.shape, dtype=dtype)
        dx_func = dx_modle._internal(m_w, m_y, m_dy).mod["main"]
        dw_func = dw_modle._internal(m_x, m_y, m_dy).mod["main"]
        dx_func = run_infer_type(dx_func)
        dw_func = run_infer_type(dw_func)
        dx_checked_type = FuncType([w_ty, y_ty, dy_ty], x_ty)
        dw_checked_type = FuncType([x_ty, y_ty, dy_ty], w_ty)
        check_type(dx_func, dx_checked_type)
        check_type(dw_func, dw_checked_type)


# pylint: disable=no-self-use, too-many-arguments
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("data_shape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("ceil", [True, False])
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.max_pool2d, torch.nn.functional.max_pool2d],
        [raf._op.sym.avg_pool2d, torch.nn.functional.avg_pool2d],
    ],
)
def test_pool2d(dtype, data_shape, kernel, stride, padding, funcs, ceil):
    if (data_shape[2] + 2 * padding - kernel) % stride != 0 and ceil:
        pytest.skip(
            """pytorch have different implementation to tvm on one side padding when the
                    stride can not fully divide the after padding shape on ceilling mode"""
        )
    raf_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class Pool2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, kernel=kernel, stride=stride, padding=padding, ceil_mode=ceil)

    model = Pool2D()
    # forward
    m_x, t_x = randn_torch(data_shape, dtype=dtype)
    m_x.requires_grad = True
    t_x.requires_grad = True
    m_y = model(m_x)
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], checked_type)
    # backward
    _, t_dy = randn_torch(m_y.shape, dtype=dtype)
    t_y.backward(t_dy)
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    dy_ty = TensorType(t_dy.shape, dtype=dtype)
    dx_ty = TensorType(t_x.grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    checked_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], checked_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "dimension",
    [
        ((2, 3), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("pad_value", [0, 2])
@pytest.mark.parametrize("pad_mode", ["constant"])
def test_pad(dtype, dimension, pad_value, pad_mode):
    shape, pad_width = dimension

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x):  # pylint: disable=no-self-use
            return raf.pad(m_x, pad_width, pad_value, pad_mode)

    m_x, t_x = randn_torch(shape, dtype=dtype)
    model = TestModel()
    m_func = model._internal(m_x).mod["main"]
    m_func = run_infer_type(m_func)
    t_y = torch.nn.functional.pad(t_x, pad_width, pad_mode, pad_value)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    y_ty = TensorType(t_y.shape, dtype=dtype)
    checked_type = FuncType([x_ty], y_ty)
    check_type(m_func, checked_type)


if __name__ == "__main__":
    pytest.main([__file__])
