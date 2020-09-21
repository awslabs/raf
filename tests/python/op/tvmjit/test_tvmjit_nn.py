import numpy as np
import pytest
import torch
import torch.nn.functional as F
import mnm


def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


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


def check_torch(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_batch_matmul(b, n, m, k, ctx):
    # pylint: disable=too-many-locals
    class BatchMatmul(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            return mnm.batch_matmul(m_a, m_b)
    # check forward
    model = BatchMatmul()
    m_a, n_a = randn((b, m, k), ctx=ctx)
    m_b, n_b = randn((b, n, k), ctx=ctx)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    n_c = np.matmul(n_a, np.transpose(n_b, (0, 2, 1)))
    check(m_c, n_c)
    # check backward
    m_dy, n_dy = randn(m_c.shape, ctx=ctx)
    m_c.backward(m_dy)
    n_dyt = np.transpose(n_dy, (0, 2, 1))
    check(m_a.grad, np.matmul(n_dy, n_b))
    check(m_b.grad, np.matmul(n_dyt, n_a))



@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_dense(n, m, k, ctx):
    # pylint: disable=no-member
    class Dense(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            return mnm.dense(m_a, m_b)
    # check forward
    model = Dense()
    m_a, n_a = randn((m, k), ctx=ctx)
    m_b, n_b = randn((n, k), ctx=ctx)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    n_c = np.matmul(n_a, np.transpose(n_b))
    check(m_c, n_c)
    # check backward
    m_dy, n_dy = randn(m_c.shape, ctx=ctx)
    m_c.backward(m_dy)
    n_dyt = np.transpose(n_dy, (1, 0))
    check(m_a.grad, np.matmul(n_dy, n_b))
    check(m_b.grad, np.matmul(n_dyt, n_a))


# pylint: disable=no-member
# pylint: disable=no-self-use
# pylint: disable=protected-access
@pytest.mark.parametrize("ctx", get_ctx_list())
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
        [mnm._op.sym.softmax, torch.nn.Softmax],
    ])
def test_unary_with_axis(shape, axis, funcs, ctx):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return m_func(x, axis=axis)
    m_func, t_func = funcs
    model = TestModel()
    # check forward
    m_x, t_x = randn_torch(shape, ctx=ctx)
    m_x.requires_grad = True
    if not -len(shape) <= axis < len(shape):
        with pytest.raises(ValueError):
            m_y = model(m_x)
        return
    m_y = model(m_x)
    t_m = t_func(axis)
    t_y = t_m(t_x)
    check_torch(m_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(shape, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x.grad, t_x.grad)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    (5, 4, 6, 9),
    (6, 5, 7, 10),
    (12, 32, 6, 8),
    (3, 7, 9)
])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("eps", [1e-05, 2e-05])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_layer_norm(ctx, shape, axis, eps, dtype):
    # pylint: disable=too-many-locals
    # pylint: disable=import-outside-toplevel
    # pylint: disable=attribute-defined-outside-init
    import mxnet as mx
    class LayerNorm(mnm.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @mnm.model.trace
        def forward(self, x):
            return mnm.layer_norm(x, axis=self._axis, eps=self._eps)
    m_model = LayerNorm(axis, eps)
    m_model.to(ctx=ctx, dtype=dtype)
    mx_model = mx.gluon.nn.LayerNorm(axis=axis, epsilon=eps, center=False, scale=False)
    mx_model.initialize(ctx=mx.cpu(0))
    m_x, n_x = randn(shape, ctx=ctx, dtype=dtype)
    mx_x = mx.nd.array(n_x)
    m_x.requires_grad = True
    mx_x.attach_grad()
    # check forward
    m_y = m_model(m_x)
    m_dy, n_dy = randn(m_y.shape, ctx=ctx, dtype=dtype)
    mx_dy = mx.nd.array(n_dy)
    with mx.autograd.record():
        mx_y = mx_model(mx_x)
        mx_y.backward(mx_dy)
    check(m_y, mx_y.asnumpy())
    # check backward
    m_y.backward(m_dy)
    check(m_x.grad, mx_x.grad.asnumpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d(ctx, dtype, xshape, wshape, stride, dilation, padding):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # N.B.: NCHW + OIHW
    # forward
    class Conv2D(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, w):  # pylint: disable=no-self-use
            return mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = Conv2D()
    # forward
    m_x, t_x = randn_torch(xshape, std=0.001, ctx=ctx, dtype=dtype)
    m_w, t_w = randn_torch(wshape, std=0.01, ctx=ctx, dtype=dtype)
    m_x.requires_grad = True
    m_w.requires_grad = True
    m_y = model(m_x, m_w)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    check_torch(m_y, t_y, rtol=1e-4, atol=1e-4)
    # backward
    m_dy, t_dy = randn_torch(t_y.shape, ctx=ctx, dtype=dtype)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x.grad, t_x.grad, rtol=1e-4, atol=1e-4)
    check_torch(m_w.grad, t_w.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_matmul(ctx, dtype, n, k, m, transpose_a, transpose_b):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
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
    m_a, t_a = randn_torch((n, k) if not transpose_a else (k, n), ctx=ctx, dtype=dtype)
    m_b, t_b = randn_torch((k, m) if not transpose_b else (m, k), ctx=ctx, dtype=dtype)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    t_c = torch.matmul(t_a.T if transpose_a else t_a, t_b.T if transpose_b else t_b) # pylint: disable=no-member
    check_torch(m_c, t_c, rtol=1e-4, atol=1e-4)
    # backward
    m_dc, t_dc = randn_torch(m_c.shape, ctx=ctx, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check_torch(m_a.grad, t_a.grad, rtol=1e-4, atol=1e-4)
    check_torch(m_b.grad, t_b.grad, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
