# pylint: disable=too-many-locals, too-many-arguments
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mnm


def randn(shape, *, ctx="cuda", dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    t_x = torch.tensor(x, requires_grad=True, device=ctx)  # pylint: disable=not-callable
    return m_x, t_x


def randn_pos(shape, *, ctx="cuda", dtype="float32", std=1.0):
    x = np.abs(np.random.randn(*shape)) * std + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    t_x = torch.tensor(x, requires_grad=True, device=ctx)  # pylint: disable=not-callable
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mnm_conv2d(xshape, wshape, stride, dilation, padding, dtype):
    # pylint: disable=too-many-locals
    # N.B.: NCHW + OIHW
    # forward
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, w):  # pylint: disable=no-self-use
            return mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = TestModel()
    # forward
    m_x, t_x = randn(xshape, std=0.001, dtype=dtype)
    m_w, t_w = randn(wshape, std=0.01, dtype=dtype)
    m_x.requires_grad = True
    m_w.requires_grad = True
    m_y = model(m_x, m_w)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    rtol = 1e-4 if dtype == "float32" else 3e-2
    atol = 1e-4 if dtype == "float32" else 3e-2
    check(m_y, t_y, rtol=rtol, atol=atol)
    # backward
    m_dy, t_dy = randn(t_y.shape, dtype=dtype)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=rtol, atol=atol)
    check(m_w.grad, t_w.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [
    [],
    [3],
    [3, 2],
    [3, 2, 5],
    [3, 2, 5, 8],
    [3, 2, 5, 8, 4],
    [3, 2, 5, 8, 4, 7],
])
@pytest.mark.parametrize(
    "funcs",
    [
        # pylint: disable=no-member,protected-access
        [mnm._op.sym.relu, torch.relu],
        [mnm._op.sym.tanh, torch.tanh],
        [mnm._op.sym.sigmoid, torch.sigmoid],
        # pylint: enable=no-member,protected-access
    ])
def test_mnm_unary(shape, funcs):
    mnm_fwd, torch_fwd = funcs

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm_fwd(x)

    model = TestModel()
    # forward
    m_x, t_x = randn(shape)
    m_x.requires_grad = True
    m_y = model(m_x)
    t_y = torch_fwd(t_x)
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(shape)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
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
        # pylint: disable=no-member,protected-access
        [mnm._op.sym.softmax, torch.softmax],
        [mnm._op.sym.log_softmax, torch.log_softmax],
        # pylint: enable=no-member,protected-access
    ])
def test_mnm_unary_with_axis(shape, axis, funcs):
    mnm_fwd, torch_fwd = funcs

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm_fwd(x, axis=axis)

    model = TestModel()
    # forward
    m_x, t_x = randn(shape)
    m_x.requires_grad = True
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_y = model(m_x)
        return
    m_y = model(m_x)
    t_y = torch_fwd(t_x, dim=axis)  # pylint: disable=no-member
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(shape)
    t_y.backward(t_dy)
    m_y.backward(m_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize(
    "funcs",
    [
        # pylint: disable=no-member,protected-access
        [mnm._op.sym.max_pool2d, torch.nn.functional.max_pool2d],
        [mnm._op.sym.avg_pool2d, torch.nn.functional.avg_pool2d],
        # pylint: enable=no-member,protected-access
    ])
def test_mnm_pool2d(kernel, stride, padding, funcs):
    mnm_fwd, torch_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm_fwd(x, kernel=kernel, stride=stride, padding=padding)

    model = TestModel()
    # forward
    m_x, t_x = randn([8, 3, 32, 32])
    m_x.requires_grad = True
    m_y = model(m_x)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(m_y.shape)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_mnm_batch_norm_infer(shape, momentum, eps):
    stats_shape = [shape[1]]
    m_x, t_x = randn(shape)
    m_m, t_m = randn(stats_shape)
    m_v, t_v = randn_pos(stats_shape)
    m_w, t_w = randn(stats_shape)
    m_b, t_b = randn(stats_shape)
    t_m.requires_grad = False
    t_v.requires_grad = False

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):  # pylint: disable=no-self-use
            return mnm.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)

    model = TestModel()
    m_y = model(m_x, m_m, m_v, m_w, m_b)
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, False,
                                         momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_mnm_batch_norm_train(shape, momentum, eps):
    stats_shape = [shape[1]]
    m_x, t_x = randn(shape)
    m_m, t_m = randn(stats_shape)
    m_v, t_v = randn_pos(stats_shape)
    m_w, t_w = randn(stats_shape)
    m_b, t_b = randn(stats_shape)
    t_m.requires_grad = False
    t_v.requires_grad = False
    m_x.requires_grad = True
    m_w.requires_grad = True
    m_b.requires_grad = True

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):  # pylint: disable=no-self-use
            result = mnm.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            return result[0]

    # forward
    model = TestModel()
    m_y = model(m_x, m_m, m_v, m_w, m_b)
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, True, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_m, t_m, rtol=1e-4, atol=1e-4)
    check(m_v, t_v, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn(shape)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad, rtol=1e-4, atol=1e-4)
    check(m_w.grad, t_w.grad, rtol=1e-4, atol=1e-4)
    check(m_b.grad, t_b.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mnm_matmul(n, k, m, transpose_a, transpose_b, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            mnm_op = [[mnm.matmul, mnm.matmul_nt],
                      [mnm.matmul_tn, mnm.matmul_tt]]
            mnm_op = mnm_op[transpose_a][transpose_b]
            return mnm_op(m_a, m_b)
    # forward
    model = TestModel()
    m_a, t_a = randn((n, k) if not transpose_a else (k, n), dtype=dtype)
    m_b, t_b = randn((k, m) if not transpose_b else (m, k), dtype=dtype)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    t_c = torch.matmul(t_a.T if transpose_a else t_a, t_b.T if transpose_b else t_b) # pylint: disable=no-member
    rtol = 1e-4 if dtype == "float32" else 2e-3
    atol = 1e-4 if dtype == "float32" else 2e-3
    check(m_c, t_c, rtol=rtol, atol=atol)
    # backward
    m_dc, t_dc = randn(m_c.shape, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad, rtol=rtol, atol=atol)
    check(m_b.grad, t_b.grad, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
