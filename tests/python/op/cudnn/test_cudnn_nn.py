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
    t_x = torch.tensor(x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
def test_mnm_conv2d(stride, dilation, padding):
    # N.B.: NCHW + OIHW
    # forward
    m_x, t_x = randn([8, 3, 32, 32], std=0.001)
    m_w, t_w = randn([16, 3, 3, 3], std=0.01)
    m_y = mnm.conv2d(m_x, m_w, stride=stride, dilation=dilation, padding=padding)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    m_dy, t_dy = randn(m_y.shape)
    # backward
    m_dx = mnm.conv2d_dx(m_w, m_y, m_dy, shape=m_x.shape, stride=stride,
                         padding=padding, dilation=dilation, groups=1)
    m_dw = mnm.conv2d_dw(m_x, m_y, m_dy, shape=m_w.shape, stride=stride,
                         padding=padding, dilation=dilation, groups=1)
    t_y.backward(t_dy)
    t_dx = t_x.grad
    t_dw = t_w.grad
    check(m_dx, t_dx, rtol=1e-4, atol=1e-4)
    check(m_dw, t_dw, rtol=1e-4, atol=1e-4)


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
@pytest.mark.parametrize("funcs", [
    # pylint: disable=no-member
    [mnm.relu, mnm.relu_dx, torch.relu],
    [mnm.tanh, mnm.tanh_dx, torch.tanh],
    [mnm.sigmoid, mnm.sigmoid_dx, torch.sigmoid],
    # pylint: enable=no-member
])
def test_mnm_unary(shape, funcs):
    mnm_fwd, mnm_bwd, torch_fwd = funcs
    # forward
    m_x, t_x = randn(shape)
    m_y = mnm_fwd(m_x)
    t_y = torch_fwd(t_x)
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(shape)
    t_y.backward(t_dy)
    m_dx = mnm_bwd(m_x, m_y, m_dy)
    t_dx = t_x.grad
    check(m_dx, t_dx)


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
@pytest.mark.parametrize("funcs", [
    # pylint: disable=no-member
    [mnm.softmax, mnm.softmax_dx, torch.softmax],
    [mnm.log_softmax, mnm.log_softmax_dx, torch.log_softmax],
    # pylint: enable=no-member
])
def test_mnm_unary_with_axis(shape, axis, funcs):
    mnm_fwd, mnm_bwd, torch_fwd = funcs
    # forward
    m_x, t_x = randn(shape)
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_y = mnm.softmax(m_x, axis=axis)
        return
    m_y = mnm_fwd(m_x, axis=axis)
    t_y = torch_fwd(t_x, dim=axis)  # pylint: disable=no-member
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(shape)
    t_y.backward(t_dy)
    m_dx = mnm_bwd(m_x, m_y, m_dy, axis)
    t_dx = t_x.grad
    check(m_dx, t_dx)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("funcs", [
    # pylint: disable=no-member
    [mnm.max_pool2d, mnm.max_pool2d_dx, torch.nn.functional.max_pool2d],
    [mnm.avg_pool2d, mnm.avg_pool2d_dx, torch.nn.functional.avg_pool2d],
    # pylint: enable=no-member
])
def test_mnm_pool2d(kernel, stride, padding, funcs):
    mnm_fwd, mnm_bwd, torch_fwd = funcs
    if padding > kernel // 2:
        return
    # forward
    m_x, t_x = randn([8, 3, 32, 32])
    m_y = mnm_fwd(m_x, kernel=kernel, stride=stride, padding=padding)
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    check(m_y, t_y)
    # backward
    m_dy, t_dy = randn(m_y.shape)
    m_dx = mnm_bwd(m_x, m_y, m_dy, kernel=kernel, stride=stride, padding=padding,
                   dilation=1, ceil_mode=False, include_pad=True)
    t_y.backward(t_dy)
    t_dx = t_x.grad
    check(m_dx, t_dx)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_mnm_batch_norm_infer(shape, momentum, eps):  # pylint: disable=too-many-locals
    stats_shape = [shape[1]]
    m_x, t_x = randn(shape)
    m_m, t_m = randn(stats_shape)
    m_v, t_v = randn(stats_shape)
    m_w, t_w = randn(stats_shape)
    m_b, t_b = randn(stats_shape)
    t_m.requires_grad = False
    t_v.requires_grad = False
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, False, momentum, eps)
    m_y = mnm.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_mnm_batch_norm_train(shape, momentum, eps):  # pylint: disable=too-many-locals
    stats_shape = [shape[1]]
    # forward
    m_x, t_x = randn(shape)
    m_m, t_m = randn(stats_shape)
    m_v, t_v = randn(stats_shape)
    m_w, t_w = randn(stats_shape)
    m_b, t_b = randn(stats_shape)
    t_m.requires_grad = False
    t_v.requires_grad = False
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, True, momentum, eps)
    m_y, m_m, m_v = mnm.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_m, t_m, rtol=1e-4, atol=1e-4)
    check(m_v, t_v, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn(shape)
    m_dx, m_dw, m_db = mnm.batch_norm_train_dxwb(m_dy, m_x, m_w, m_b, eps=eps)
    t_y.backward(t_dy)
    t_dx, t_dw, t_db = t_x.grad, t_w.grad, t_b.grad
    check(m_dx, t_dx, rtol=1e-4, atol=1e-4)
    check(m_dw, t_dw, rtol=1e-4, atol=1e-4)
    check(m_db, t_db, rtol=1e-4, atol=1e-4)


# TODO(@were): bias cannot be supported in CUDNN without additional mechanisms
@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("axis", range(-8, 8))
def test_mnm_bias_add(axis): #pylint: disable=unused-argument
    return
    # pylint: disable=unreachable
    ndims = 3
    shape = [2] * ndims
    x = np.arange(1, 2 ** ndims + 1).astype('float32')
    b = np.arange(1, ndims + 1).astype('float32')
    x.shape = shape
    # n_y = x + np.expand_dims(b, axis)
    x = mnm.array(x, ctx='cuda')
    b = mnm.array(b, ctx='cuda')
    # y = mnm.bias_add(x, b, axis=axis)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_mnm_matmul(n, k, m, transpose_a, transpose_b):
    a = np.arange(1, n * k + 1).astype('float32')
    a.shape = (n, k) if not transpose_a else (k, n)
    b = np.arange(1, k * m + 1).astype('float32')
    b.shape = (k, m) if not transpose_b else (m, k)
    n_c = np.dot(a if not transpose_a else a.T, b if not transpose_b else b.T)
    a = mnm.array(a, ctx='cuda')
    b = mnm.array(b, ctx='cuda')
    m_c = mnm.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    np.testing.assert_allclose(m_c.asnumpy(), n_c, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main()
