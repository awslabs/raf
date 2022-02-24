# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals, no-self-use, line-too-long, attribute-defined-outside-init
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import raf
from raf.testing import (
    randint,
    randn,
    numpy,
    get_testable_devices,
    randn_torch,
    with_seed,
    check,
    run_vm_model,
    with_dialect,
)
from raf.model.trace import trace_mutate_attr


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("broadcast", ["none", "a", "b"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_batch_matmul(device, dtype, b, n, k, m, broadcast, transpose_a, transpose_b):
    # pylint: disable=too-many-arguments, invalid-name
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):
            raf_op = [
                [raf.batch_matmul, raf.batch_matmul_nt],
                [raf.batch_matmul_tn, raf.batch_matmul_tt],
            ]
            raf_op = raf_op[transpose_a][transpose_b]
            return raf_op(m_a, m_b)

    b1 = b
    b2 = b
    if broadcast == "a":
        b1 = 1
    elif broadcast == "b":
        b2 = 1
    # forward
    model = TestModel()
    m_a, t_a = randn_torch(
        (b1, n, k) if not transpose_a else (b1, k, n),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    m_b, t_b = randn_torch(
        (b2, k, m) if not transpose_b else (b2, m, k),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b], disable_fusion=True)

    t_at = torch.transpose(t_a, 1, 2) if transpose_a else t_a
    t_bt = torch.transpose(t_b, 1, 2) if transpose_b else t_b
    t_c = torch.matmul(t_at, t_bt)  # pylint: disable=no-member
    check(m_c, t_c, rtol=1e-4, atol=1e-4)
    check(v_c, t_c, rtol=1e-4, atol=1e-4)
    # backward
    m_dc, t_dc = randn_torch(m_c.shape, device=device, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad, rtol=1e-4, atol=1e-4)
    check(m_b.grad, t_b.grad, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("k", [1, 4])
def test_dense(n, m, k, device):
    # pylint: disable=no-member
    class Dense(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):
            return raf.dense(m_a, m_b)

    # check forward
    model = Dense()
    m_a, n_a = randn((m, k), device=device)
    m_b, n_b = randn((n, k), device=device)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b], disable_fusion=True)
    n_c = np.matmul(n_a, np.transpose(n_b))
    check(m_c, n_c)
    check(v_c, n_c)
    # check backward
    m_dy, n_dy = randn(m_c.shape, device=device)
    m_c.backward(m_dy)
    n_dyt = np.transpose(n_dy, (1, 0))
    check(m_a.grad, np.matmul(n_dy, n_b))
    check(m_b.grad, np.matmul(n_dyt, n_a))


# pylint: disable=no-member
# pylint: disable=protected-access
# TODO: Currently TVM fails to schedule softmax or softmax_dx in certain cases
@pytest.mark.skip
@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.softmax, torch.softmax],
    ],
)
def test_unary_with_axis(device, dtype, shape, axis, funcs):
    raf_fwd, torch_fwd = funcs

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(shape, device=device, dtype=dtype, requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x], disable_fusion=True)
    t_y = torch_fwd(t_x, dim=axis)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, device=device, dtype=dtype)
    t_y.backward(t_dy)
    m_y.backward(m_dy)
    check(m_x.grad, t_x.grad)


# pylint: disable=no-member
# pylint: disable=protected-access
@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("shape", [[3, 2], [1, 3]])
def test_log_softmax(device, dtype, shape):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf._op.sym.log_softmax(x)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(shape, device=device, dtype=dtype, requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x], disable_fusion=True)
    t_y = torch.log_softmax(t_x, dim=-1)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, device=device, dtype=dtype)
    t_y.backward(t_dy)
    m_y.backward(m_dy)
    check(m_x.grad, t_x.grad)


# pylint: disable=too-many-arguments
@with_dialect("tvm")
@with_seed(0)
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [(5, 4, 6, 9), (3, 7, 9)])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("eps", [1e-05, 2e-05])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("learnable_affine_transform", [False, True])
def test_layer_norm(device, shape, axis, eps, dtype, learnable_affine_transform):
    # pylint: disable=import-outside-toplevel
    import mxnet as mx

    class LayerNorm(raf.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @raf.model.trace
        def forward(self, *inputs):
            return raf.layer_norm(*inputs, axis=self._axis, eps=self._eps)

    m_model = LayerNorm(axis, eps)
    m_model.to(device=device, dtype=dtype)
    mx_model = mx.gluon.nn.LayerNorm(
        axis=axis, epsilon=eps, center=learnable_affine_transform, scale=learnable_affine_transform
    )
    mx_model.initialize(ctx=mx.cpu(0))

    m_x, n_x = randn(shape, device=device, dtype=dtype)
    mx_x = mx.nd.array(n_x)
    m_x.requires_grad = True
    mx_x.attach_grad()

    if learnable_affine_transform:
        m_scale, n_scale = randn([shape[axis]], device=device, dtype=dtype)
        m_bias, n_bias = randn([shape[axis]], device=device, dtype=dtype)
        m_scale.requires_grad = True
        m_bias.requires_grad = True
        mx_scale = mx.nd.array(n_scale)
        mx_bias = mx.nd.array(n_bias)
        mx_scale.attach_grad()
        mx_bias.attach_grad()
        mx_model.gamma.set_data(mx_scale)
        mx_model.beta.set_data(mx_bias)
        # check forward
        m_y = m_model(m_x, m_scale, m_bias)
        v_y = run_vm_model(m_model, device, [m_x, m_scale, m_bias], disable_fusion=True)
    else:
        m_y = m_model(m_x)
        v_y = run_vm_model(m_model, device, [m_x], disable_fusion=True)

    m_dy, n_dy = randn(m_y.shape, device=device, dtype=dtype)
    mx_dy = mx.nd.array(n_dy)
    with mx.autograd.record():
        mx_y = mx_model(mx_x)
        mx_y.backward(mx_dy)

    check(m_y, mx_y, rtol=1e-4, atol=1e-4)
    check(v_y, mx_y, rtol=1e-4, atol=1e-4)
    # check backward
    m_y.backward(m_dy)
    check(m_x.grad, mx_x.grad, rtol=1e-4, atol=1e-4)
    if learnable_affine_transform:
        check(m_scale.grad, mx_model.gamma.grad(), rtol=1e-4, atol=1e-4)
        check(m_bias.grad, mx_model.beta.grad(), rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shapes",
    [
        ((4, 256, 32, 32), (64, 256, 1, 1)),
        ((8, 3, 32, 32), (16, 3, 3, 3)),
    ],
)
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d(device, dtype, shapes, stride, dilation, padding):
    # pylint: disable=too-many-arguments
    # N.B.: NCHW + OIHW
    # forward
    class Conv2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            return raf.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = Conv2D()
    # forward
    xshape, wshape = shapes
    m_x, t_x = randn_torch(xshape, std=0.001, device=device, dtype=dtype, requires_grad=True)
    m_w, t_w = randn_torch(wshape, std=0.01, device=device, dtype=dtype, requires_grad=True)
    m_y = model(m_x, m_w)
    v_y = run_vm_model(model, device, [m_x, m_w], disable_fusion=True)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(v_y, t_y, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn_torch(t_y.shape, device=device, dtype=dtype)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=1e-4, atol=1e-4)
    check(m_w.grad, t_w.grad, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shapes",
    [
        ((4, 256, 32, 32), (256, 64, 4, 4)),
        ((8, 3, 32, 32), (3, 16, 3, 3)),
    ],
)
@pytest.mark.parametrize(
    "stride_output_padding",
    [
        (1, 0),
        (2, 1),
    ],
)
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d_trans(device, dtype, shapes, stride_output_padding, dilation, padding):
    # pylint: disable=too-many-arguments
    # N.B.: NCHW + OIHW
    # forward
    class Conv2DTrans(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            return raf.conv2d_transpose(
                x,
                w,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=1,
            )  # pylint: disable=line-too-long

    model = Conv2DTrans()
    # forward
    stride, output_padding = stride_output_padding
    xshape, wshape = shapes
    m_x, t_x = randn_torch(xshape, std=0.001, device=device, dtype=dtype, requires_grad=True)
    m_w, t_w = randn_torch(wshape, std=0.01, device=device, dtype=dtype, requires_grad=True)
    t_y = F.conv_transpose2d(
        t_x, t_w, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding
    )
    m_y = model(m_x, m_w)
    v_y = run_vm_model(model, device, [m_x, m_w], disable_fusion=True)

    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(v_y, t_y, rtol=1e-4, atol=1e-4)

    # backward
    m_dy, t_dy = randn_torch(t_y.shape, device=device, dtype=dtype)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=1e-4, atol=1e-4)
    check(m_w.grad, t_w.grad, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d_nhwc(device, dtype, xshape, wshape, stride, dilation, padding):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # N.B.: NHWC + HWIO
    # forward
    class Conv2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            x = raf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            w = raf.transpose(w, (2, 3, 1, 0))  # OIHW -> HWIO
            conv = raf.conv2d(
                x,
                w,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                layout="NHWC",
                kernel_layout="HWIO",
                out_layout="NHWC",
            )
            # NHWC -> NCHW
            return raf.transpose(conv, (0, 3, 1, 2))

    model = Conv2D()
    m_x, t_x = randn_torch(xshape, std=0.001, device=device, dtype=dtype)
    m_w, t_w = randn_torch(wshape, std=0.01, device=device, dtype=dtype)
    # forward only for NHWC
    m_y = model(m_x, m_w)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("xshape", [(3, 3, 4, 4), (8, 3, 32, 32)])
@pytest.mark.parametrize("axis", [-1, 1])
def test_bias_add(xshape, dtype, device, axis):
    class BiasAdd(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, bias):
            return raf.bias_add(x, bias, axis=axis)

    model = BiasAdd()
    bshape = (xshape[axis],)
    dim = len(xshape)
    m_x, n_x = randn(xshape, dtype=dtype, device=device, requires_grad=True)
    m_bias, n_bias = randn(bshape, dtype=dtype, device=device, requires_grad=True)
    m_dy, n_dy = randn(xshape, dtype=dtype, device=device)
    # check forward
    n_axis = (axis % dim + dim) % dim
    n_bias = np.reshape(n_bias, tuple([bshape[0]] + [1] * (dim - n_axis - 1)))
    n_y = np.add(n_x, n_bias)
    m_y = model(m_x, m_bias)
    check(m_y, n_y)
    # check backward
    m_y.backward(m_dy)
    n_dx = np.reshape(n_dy, xshape)
    axes = list(range(len(xshape)))
    axes.pop(n_axis)
    n_db = np.sum(n_dy, axis=tuple(axes))
    check(m_x.grad, n_dx)
    check(m_bias.grad, n_db, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", ["cpu"])
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
def test_pool2d(device, dtype, data_shape, kernel, stride, padding, funcs, ceil):
    if (data_shape[2] + 2 * padding - kernel) % stride != 0 and ceil:
        pytest.skip(
            """pytorch have different implementation to tvm on one side padding when the
                    stride can not fully divide the after padding shape on ceilling mode"""
        )
    # TODO(@XIAO-XIA): complement test case when device=cuda
    raf_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, kernel=kernel, stride=stride, padding=padding, ceil_mode=ceil)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(data_shape, dtype=dtype, device=device, requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x], disable_fusion=True)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil)
    check(m_y, t_y)
    check(v_y, t_y)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("data_shape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("out_shape", [(1, 1), (4, 4)])
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.adaptive_max_pool2d, torch.nn.functional.adaptive_max_pool2d],
        [raf._op.sym.adaptive_avg_pool2d, torch.nn.functional.adaptive_avg_pool2d],
    ],
)
def test_adaptive_pool2d(device, dtype, data_shape, out_shape, funcs):
    raf_fwd, torch_fwd = funcs

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, out_shape)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(data_shape, dtype=dtype, device=device, requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x], disable_fusion=True)
    t_y = torch_fwd(t_x, out_shape)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(m_y.shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@with_dialect("tvm")
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("data_shape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.max_pool2d, torch.nn.functional.max_pool2d],
        [raf._op.sym.avg_pool2d, torch.nn.functional.avg_pool2d],
    ],
)
def test_pool2d_nhwc(device, dtype, data_shape, kernel, stride, padding, funcs):
    # TODO(yzhliu): complement test case when device=cuda
    # pylint: disable=too-many-locals, too-many-arguments
    raf_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            pool = raf_fwd(x, kernel=kernel, stride=stride, padding=padding, layout="NHWC")
            # NHWC -> NCHW
            return raf.transpose(pool, (0, 3, 1, 2))

    model = TestModel()
    m_x, t_x = randn_torch(data_shape, dtype=dtype, device=device, requires_grad=False)
    # forward only for NHWC layout
    m_y = model(m_x)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    check(m_y, t_y)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_matmul(device, dtype, n, k, m, transpose_a, transpose_b):
    # pylint: disable=too-many-arguments
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):
            raf_op = [[raf.matmul, raf.matmul_nt], [raf.matmul_tn, raf.matmul_tt]]
            raf_op = raf_op[transpose_a][transpose_b]
            return raf_op(m_a, m_b)

    # forward
    model = TestModel()
    m_a, t_a = randn_torch(
        (n, k) if not transpose_a else (k, n), device=device, dtype=dtype, requires_grad=True
    )
    m_b, t_b = randn_torch(
        (k, m) if not transpose_b else (m, k), device=device, dtype=dtype, requires_grad=True
    )
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b], disable_fusion=True)
    t_c = torch.matmul(
        t_a.T if transpose_a else t_a, t_b.T if transpose_b else t_b
    )  # pylint: disable=no-member
    check(m_c, t_c, rtol=1e-4, atol=1e-4)
    check(v_c, t_c, rtol=1e-4, atol=1e-4)
    # backward
    m_dc, t_dc = randn_torch(m_c.shape, device=device, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad, rtol=1e-4, atol=1e-4)
    check(m_b.grad, t_b.grad, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
def test_raf_batch_norm_infer(shape, momentum, eps, device):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, device=device)
    m_m, t_m = randn_torch(stats_shape, device=device)
    m_v, t_v = randn_torch(stats_shape, device=device, positive=True)
    m_w, t_w = randn_torch(stats_shape, device=device)
    m_b, t_b = randn_torch(stats_shape, device=device)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):  # pylint: disable=too-many-arguments
            return raf.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)

    model = TestModel()
    m_y = model(m_x, m_m, m_v, m_w, m_b)
    v_y = run_vm_model(model, device, [m_x, m_m, m_v, m_w, m_b], disable_fusion=True)
    t_y = F.batch_norm(t_x, t_m, t_v, t_w, t_b, False, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(v_y, t_y, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
@with_seed(0)
def test_raf_batch_norm_train(shape, momentum, eps, device):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, device=device, requires_grad=True)
    m_m, t_m = randn_torch(stats_shape, device=device)
    m_v, t_v = randn_torch(stats_shape, device=device, positive=True)
    m_w, t_w = randn_torch(stats_shape, device=device, requires_grad=True)
    m_b, t_b = randn_torch(stats_shape, device=device, requires_grad=True)
    np_m = m_m.numpy()
    np_v = m_v.numpy()

    class TestModel(raf.Model):
        def build(self, m_m, m_v):
            self.m_m = m_m
            self.m_v = m_v

        @raf.model.trace
        def forward(self, m_x, m_w, m_b):  # pylint: disable=too-many-arguments
            result = raf.batch_norm_train(m_x, self.m_m, self.m_v, m_w, m_b, momentum, eps)
            trace_mutate_attr(self, "m_m", result[1])
            trace_mutate_attr(self, "m_v", result[2])
            return result[0]

    # forward
    model = TestModel(m_m, m_v)
    m_y = model(m_x, m_w, m_b)
    t_y = F.batch_norm(t_x, t_m, t_v, t_w, t_b, True, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_m, t_m, rtol=1e-4, atol=1e-4)
    check(m_v, t_v, rtol=1e-4, atol=1e-4)
    # forward vm
    model.m_m = raf.array(np_m, device=device)
    model.m_v = raf.array(np_v, device=device)
    v_y = run_vm_model(model, device, [m_x, m_w, m_b], disable_fusion=True)[0]
    check(v_y, t_y, rtol=1e-4, atol=1e-4)
    check(model.m_m, t_m, rtol=1e-4, atol=1e-4)
    check(model.m_v, t_v, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn_torch(shape, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=1e-4, atol=1e-4)
    check(m_w.grad, t_w.grad, rtol=1e-4, atol=1e-4)
    check(m_b.grad, t_b.grad, rtol=1e-4, atol=1e-4)


@with_dialect("tvm")
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "dimension",
    [
        ((2, 3), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("pad_value", [0, 2])
@pytest.mark.parametrize("pad_mode", ["constant"])
def test_pad(device, dtype, dimension, pad_value, pad_mode):
    shape, pad_width = dimension

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x):
            return raf.pad(m_x, pad_width, pad_value, pad_mode)

    m_x, t_x = randn_torch(shape, device=device, dtype=dtype)
    model = TestModel()
    m_y = model(m_x)
    t_y = torch.nn.functional.pad(t_x, pad_width, pad_mode, pad_value)
    check(m_y, t_y)


@with_dialect("tvm")
@pytest.mark.parametrize("hyperparam", [(0.6, 1.2), (-0.2, 1.2)])
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [(), (1,), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32"])
def test_threshold_with_grad(hyperparam, shape, dtype, device):
    class TestModel(raf.Model):
        def build(self, threshold, value):
            self.threshold = threshold
            self.value = value

        @raf.model.trace
        def forward(self, x):
            return raf._op.sym.threshold(x, self.threshold, self.value)

    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    threshold, value = hyperparam
    model = TestModel(threshold, value)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x], disable_fusion=True)
    t_model = torch.nn.Threshold(threshold, value)
    t_y = t_model(t_x)
    # check forward
    check(m_y, t_y)
    check(v_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


# pylint: disable=protected-access
# cannot use TVM dropout op because
# TODO(@yzhliu): Currently we cannot enable dropout test on CUDA because TVM doesn't support random
#   on CUDA.
@with_dialect("tvm")
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("dropout", [0.6])
def test_raf_dropout(dropout, device):
    def check_dropout(x, y, dx=None, dy=None):
        x, y = x.numpy(), y.numpy()
        mask = y != 0
        expected = mask * x / (1 - dropout)
        check(expected, y)
        frac = np.sum(y == 0) / y.size
        assert dropout - 0.1 < frac < dropout + 0.1
        if dx is not None and dy is not None:
            dx, dy = dx.numpy(), dy.numpy()
            expected = mask / (1 - dropout) * dy
            check(expected, dx)

    class TestModel(raf.Model):
        def build(self):
            self.dropout = dropout

        @raf.model.trace
        def forward(self, x):
            return raf._contrib_dropout(x, dropout)

    shape, dtype = [128, 128], "float32"
    x, _ = randint(shape, low=10, high=20, dtype=dtype, device=device)
    x.requires_grad = True
    model = TestModel()

    m_y = model(x)[0]
    check_dropout(x, m_y)

    # TODO(yizhiliu): unify CPU and CUDNN (defined in test_cudnn_nn.py) dropout model
    v_y = run_vm_model(model, device, [x], disable_fusion=True)[0]
    check_dropout(x, v_y)

    n_y = model(x)[0]
    assert not np.array_equal(numpy(m_y), numpy(v_y))
    assert not np.array_equal(numpy(m_y), numpy(n_y))
    # backward
    dy, _ = randn_torch(shape, dtype=dtype)
    m_y.backward(dy)
    check_dropout(x, m_y, x.grad, dy)


if __name__ == "__main__":
    pytest.main([__file__])
