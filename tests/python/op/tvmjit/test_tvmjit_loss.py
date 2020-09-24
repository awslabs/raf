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
    t_x = torch.tensor(n_x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def one_hot(batch_size, num_classes, ctx="cpu", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    t_x = torch.tensor(targets, requires_grad=False)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.cpu().detach().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("c", [2, 4, 6])
def test_nll_loss(ctx, n, c):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return mnm.nll_loss(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    m_pred, t_pred = randn((n, c), ctx=ctx)
    m_true, t_true = one_hot(n, c, ctx=ctx)
    m_pred.requires_grad = True
    # forward
    t_loss = F.nll_loss(t_pred, t_true)
    m_loss = model(y_true=m_true, y_pred=m_pred)
    check(m_loss, t_loss)
    # backward
    t_loss.backward()
    m_loss.backward()
    check(m_pred.grad, t_pred.grad)


@pytest.mark.parametrize("ctx", ['cpu'])
@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("c", [2, 4, 6])
def test_cross_entropy(ctx, n, c):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return mnm.cross_entropy(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    m_pred, t_pred = randn((n, c), ctx=ctx)
    m_true, t_true = one_hot(n, c, ctx=ctx)
    m_pred.requires_grad = True
    # forward
    t_loss = F.cross_entropy(t_pred, t_true)
    m_loss = model(y_true=m_true, y_pred=m_pred)
    check(m_loss, t_loss)
    # backward
    t_loss.backward()
    m_loss.backward()
    check(m_pred.grad, t_pred.grad)


@pytest.mark.parametrize("shape", [
    (3, 16, 128, 128),
    (3, 16),
])
@pytest.mark.parametrize("ctx", get_ctx_list())
def test_broadcast_add(shape, ctx):
    m_a, t_a = randn(shape, ctx=ctx)
    n = len(shape)
    m_b, t_b = randn([shape[1]] + [1] * (n - 2), ctx=ctx)
    t_y = t_a + t_b
    m_y = mnm.add(m_a, m_b)
    check(m_y, t_y)
    m_dy, t_dy = randn(m_y.shape, ctx=ctx)
    t_y.backward(t_dy)

    def mnm_add_dx(m_dy, m_x):
        axis = mnm.get_reduce_axis(m_dy, m_x)
        keep = mnm.get_kept_dims(m_dy, m_x)
        return mnm.sum(m_dy, axis=axis, keepdims=keep)

    m_da = mnm_add_dx(m_dy, m_a)
    check(m_da, t_a.grad, atol=0.01 if n == 4 else 1e-4)
    m_db = mnm_add_dx(m_dy, m_b)
    check(m_db, t_b.grad, atol=0.01 if n == 4 else 1e-4)


@pytest.mark.parametrize("shape", [[4, 4]])
def test_sgd(shape):
    x0 = np.random.randn(*shape).astype('float32')
    dx = np.random.randn(*shape).astype('float32')
    v0 = np.random.randn(*shape).astype('float32')
    mu = np.random.randn(1)[0]  # pylint: disable=invalid-name
    learning_rate = 0.01

    n_v1 = (mu * v0 + dx)
    n_x1 = (x0 - learning_rate * n_v1)

    x0 = mnm.array(x0)
    dx = mnm.array(dx)
    v0 = mnm.array(v0)
    m_v1, m_x1 = mnm.sgd(x0, dx, v0, learning_rate, mu)

    np.testing.assert_allclose(m_v1.asnumpy(), n_v1, 1e-4, 1e-4)
    np.testing.assert_allclose(m_x1.asnumpy(), n_x1, 1e-4, 1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
