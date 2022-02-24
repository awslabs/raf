# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,attribute-defined-outside-init
# pylint: disable=no-self-use,no-member
import random
import pytest
import torch
import torch.nn.functional as F
import numpy as np

import raf
from raf.testing import randint, randn_torch, run_vm_model, check, numpy, with_seed, with_dialect
from raf._core.ndarray import ndarray


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_raf_conv2d(xshape, wshape, stride, dilation, padding, dtype):
    # N.B.: NCHW + OIHW
    # forward
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            return raf.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(xshape, device="cuda", std=0.001, dtype=dtype, requires_grad=True)
    m_w, t_w = randn_torch(wshape, device="cuda", std=0.01, dtype=dtype, requires_grad=True)
    m_y = model(m_x, m_w)
    v_y = run_vm_model(model, "cuda", [m_x, m_w])
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    rtol = 1e-4 if dtype == "float32" else 4e-2
    atol = 1e-4 if dtype == "float32" else 4e-2
    check(m_y, t_y, rtol=rtol, atol=atol)
    check(v_y, t_y, rtol=rtol, atol=atol)
    # backward
    m_dy, t_dy = randn_torch(t_y.shape, device="cuda", dtype=dtype)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=rtol, atol=atol)
    check(m_w.grad, t_w.grad, rtol=rtol, atol=atol)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.relu, torch.relu],
        [raf._op.sym.tanh, torch.tanh],
        [raf._op.sym.sigmoid, torch.sigmoid],
    ],
)
def test_raf_unary(shape, funcs):
    raf_fwd, torch_fwd = funcs

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(shape, device="cuda", requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, device="cuda")
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize("axis", range(-8, 8))
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.softmax, torch.softmax],
        [raf._op.sym.log_softmax, torch.log_softmax],
    ],
)
def test_raf_softmax(shape, axis, funcs):
    raf_fwd, torch_fwd = funcs

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(shape, device="cuda", requires_grad=True)
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_y = model(m_x)
        return
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x, dim=axis)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, device="cuda")
    t_y.backward(t_dy)
    m_y.backward(m_dy)
    check(m_x.grad, t_x.grad)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
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
def test_raf_pool2d(kernel, stride, padding, funcs):
    raf_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, kernel=kernel, stride=stride, padding=padding)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch([8, 3, 32, 32], device="cuda", requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(m_y.shape, device="cuda")
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@with_seed(0)
def test_raf_batch_norm_infer(shape, momentum, eps, dtype):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, dtype=dtype, device="cuda")
    m_m, t_m = randn_torch(stats_shape, device="cuda")
    m_v, t_v = randn_torch(stats_shape, device="cuda", positive=True)
    m_w, t_w = randn_torch(stats_shape, device="cuda")
    m_b, t_b = randn_torch(stats_shape, device="cuda")

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):
            return raf.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)

    model = TestModel()
    m_y = model(m_x, m_m, m_v, m_w, m_b)
    v_y = run_vm_model(model, "cuda", [m_x, m_m, m_v, m_w, m_b])
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, False, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(v_y, t_y, rtol=1e-4, atol=1e-4)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-6])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@with_seed(0)
def test_raf_batch_norm_train(shape, momentum, eps, dtype):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, dtype=dtype, device="cuda", requires_grad=True)
    m_mean, t_mean = randn_torch(stats_shape, device="cuda")
    m_var, t_var = randn_torch(stats_shape, device="cuda", positive=True)
    m_w, t_w = randn_torch(stats_shape, device="cuda", requires_grad=True)
    m_b, t_b = randn_torch(stats_shape, device="cuda", requires_grad=True)
    np_mean = m_mean.numpy()
    np_var = m_var.numpy()

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):
            result = raf.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            return result[0]

    # forward
    model = TestModel()
    m_y = model(m_x, m_mean, m_var, m_w, m_b)
    t_y = torch.nn.functional.batch_norm(t_x, t_mean, t_var, t_w, t_b, True, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_mean, t_mean, rtol=1e-4, atol=1e-4)
    check(m_var, t_var, rtol=1e-4, atol=1e-4)
    m_mean = raf.array(np_mean, device="cuda")
    m_var = raf.array(np_var, device="cuda")
    v_y = run_vm_model(model, "cuda", [m_x, m_mean, m_var, m_w, m_b])
    check(v_y, t_y, rtol=1e-5, atol=1e-5)
    check(m_mean, t_mean, rtol=1e-5, atol=1e-5)
    check(m_var, t_var, rtol=1e-5, atol=1e-5)

    # backward
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device="cuda")
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    rtol = 1e-5 if dtype == "float32" else 1e-3
    atol = 1e-5 if dtype == "float32" else 1e-3
    check(m_x.grad, t_x.grad, rtol=rtol, atol=atol)
    check(m_w.grad, t_w.grad, rtol=rtol, atol=atol)
    check(m_b.grad, t_b.grad, rtol=rtol, atol=atol)


@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("dropout", [0.4, 0.6])
def test_raf_dropout(dropout):
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
            self.dropout_state = ndarray.from_tensor_value(
                raf._ffi.backend.cudnn.GetDropoutState(dropout, random.getrandbits(63))
            ).to(device="cuda")

        @raf.model.trace
        def forward(self, x):
            return raf._contrib_dropout(x, dropout, self.dropout_state)

    shape, dtype = [1024, 1024], "float32"
    x, _ = randint(shape, low=10, high=20, dtype=dtype, device="cuda")
    x.requires_grad = True
    model = TestModel()
    state_0 = model.dropout_state.to(device="cuda")
    m_y = model(x)[0]
    state_1 = model.dropout_state.to(device="cuda")
    check_dropout(x, m_y)
    v_y = run_vm_model(model, "cuda", [x])[0]
    state_2 = model.dropout_state.to(device="cuda")
    check_dropout(x, v_y)
    # state updates (cudnn enforce state inplace updates)
    n_y = model(x)[0]
    assert not np.array_equal(numpy(state_0), numpy(state_1))
    assert not np.array_equal(numpy(state_1), numpy(state_2))
    assert not np.array_equal(numpy(m_y), numpy(v_y))
    assert not np.array_equal(numpy(m_y), numpy(n_y))
    # reproducible
    model.dropout_state = state_0
    r_y = model(x)[0]
    check(m_y, r_y)
    # backward
    dy, _ = randn_torch(shape, dtype=dtype, device="cuda")
    m_y.backward(dy)
    check_dropout(x, m_y, x.grad, dy)


if __name__ == "__main__":
    pytest.main([__file__])
