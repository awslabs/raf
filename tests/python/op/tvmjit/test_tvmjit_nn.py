import numpy as np
import pytest
import torch
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


def randn_torch(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
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


if __name__ == "__main__":
    pytest.main([__file__])
