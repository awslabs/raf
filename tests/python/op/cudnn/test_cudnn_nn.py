# pylint: disable=too-many-locals,too-many-arguments,protected-access
import pytest
import torch
import torch.nn.functional as F

import mnm
from mnm.testing import randn_torch, run_vm_model, check


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
    m_x, t_x = randn_torch(xshape, ctx="cuda", std=0.001, dtype=dtype, requires_grad=True)
    m_w, t_w = randn_torch(wshape, ctx="cuda", std=0.01, dtype=dtype, requires_grad=True)
    m_y = model(m_x, m_w)
    v_y = run_vm_model(model, "cuda", [m_x, m_w])
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    rtol = 1e-4 if dtype == "float32" else 4e-2
    atol = 1e-4 if dtype == "float32" else 4e-2
    check(m_y, t_y, rtol=rtol, atol=atol)
    check(v_y, t_y, rtol=rtol, atol=atol)
    # backward
    m_dy, t_dy = randn_torch(t_y.shape, ctx="cuda", dtype=dtype)
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
    m_x, t_x = randn_torch(shape, ctx="cuda", requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, ctx="cuda")
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
def test_mnm_softmax(shape, axis, funcs):
    mnm_fwd, torch_fwd = funcs

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm_fwd(x, axis=axis)

    model = TestModel()
    # forward
    m_x, t_x = randn_torch(shape, ctx="cuda", requires_grad=True)
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_y = model(m_x)
        return
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x, dim=axis)  # pylint: disable=no-member
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(shape, ctx="cuda")
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
    m_x, t_x = randn_torch([8, 3, 32, 32], ctx="cuda", requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, "cuda", [m_x])
    t_y = torch_fwd(t_x, kernel_size=kernel, stride=stride, padding=padding)
    check(m_y, t_y)
    check(v_y, t_y)
    # backward
    m_dy, t_dy = randn_torch(m_y.shape, ctx="cuda")
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
def test_mnm_batch_norm_infer(shape, momentum, eps):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, ctx="cuda")
    m_m, t_m = randn_torch(stats_shape, ctx="cuda")
    m_v, t_v = randn_torch(stats_shape, ctx="cuda", positive=True)
    m_w, t_w = randn_torch(stats_shape, ctx="cuda")
    m_b, t_b = randn_torch(stats_shape, ctx="cuda")

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):  # pylint: disable=no-self-use
            return mnm.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)

    model = TestModel()
    m_y = model(m_x, m_m, m_v, m_w, m_b)
    v_y = run_vm_model(model, "cuda", [m_x, m_m, m_v, m_w, m_b])
    t_y = torch.nn.functional.batch_norm(t_x, t_m, t_v, t_w, t_b, False,
                                         momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(v_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [[8, 8, 8, 8], [8, 8, 8, 8, 8]])
@pytest.mark.parametrize("momentum", [0.1, 0.2, 0.3, 0.4])
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5, 1e-6])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mnm_batch_norm_train(shape, momentum, eps, dtype):
    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, dtype=dtype, ctx="cuda", requires_grad=True)
    m_mean, t_mean = randn_torch(stats_shape, ctx="cuda")
    m_var, t_var = randn_torch(stats_shape, ctx="cuda", positive=True)
    m_w, t_w = randn_torch(stats_shape, ctx="cuda", requires_grad=True)
    m_b, t_b = randn_torch(stats_shape, ctx="cuda", requires_grad=True)
    np_mean = m_mean.asnumpy()
    np_var = m_var.asnumpy()

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):  # pylint: disable=no-self-use
            result = mnm.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            return result[0]

    # forward
    model = TestModel()
    m_y = model(m_x, m_mean, m_var, m_w, m_b)
    t_y = torch.nn.functional.batch_norm(t_x, t_mean, t_var, t_w, t_b, True, momentum, eps)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_mean, t_mean, rtol=1e-4, atol=1e-4)
    check(m_var, t_var, rtol=1e-4, atol=1e-4)
    m_mean = mnm.array(np_mean, ctx="cuda")
    m_var = mnm.array(np_var, ctx="cuda")
    v_y = run_vm_model(model, "cuda", [m_x, m_mean, m_var, m_w, m_b])
    check(v_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_mean, t_mean, rtol=1e-4, atol=1e-4)
    check(m_var, t_var, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn_torch(shape, dtype=dtype, ctx="cuda")
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    rtol = 1e-4 if dtype == "float32" else 1e-3
    atol = 1e-4 if dtype == "float32" else 1e-3
    check(m_x.grad, t_x.grad, rtol=rtol, atol=atol)
    check(m_w.grad, t_w.grad, rtol=rtol, atol=atol)
    check(m_b.grad, t_b.grad, rtol=rtol, atol=atol)


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
    m_a, t_a = randn_torch((n, k) if not transpose_a else (k, n),
                           dtype=dtype, ctx="cuda", requires_grad=True)
    m_b, t_b = randn_torch((k, m) if not transpose_b else (m, k),
                           dtype=dtype, ctx="cuda", requires_grad=True)
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, "cuda", [m_a, m_b])
    t_c = torch.matmul(t_a.T if transpose_a else t_a, t_b.T if transpose_b else t_b) # pylint: disable=no-member
    rtol = 1e-4 if dtype == "float32" else 2e-3
    atol = 1e-4 if dtype == "float32" else 2e-3
    check(m_c, t_c, rtol=rtol, atol=atol)
    check(v_c, t_c, rtol=rtol, atol=atol)
    # backward
    m_dc, t_dc = randn_torch(m_c.shape, dtype=dtype, ctx="cuda")
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad, rtol=rtol, atol=atol)
    check(m_b.grad, t_b.grad, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
