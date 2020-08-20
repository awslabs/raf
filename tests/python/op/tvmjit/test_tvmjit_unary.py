import numpy as np
import pytest
from scipy import special
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


def randn_pos(shape, *, ctx="cpu", dtype="float32"):
    x = np.abs(np.random.randn(*shape) + 1e-5)
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
@pytest.mark.parametrize(
    "ops",
    [
        (np.copy, mnm.copy),
        (np.ceil, mnm.ceil),
        (np.floor, mnm.floor),
        (np.cos, mnm.cos),
        (np.abs, mnm.abs),
        (np.exp, mnm.exp),
        (np.arctan, mnm.atan),
        (special.erf, mnm.erf),  # pylint: disable=no-member
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops(ops, shape, dtype, ctx):
    n_op, m_op = ops
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_x = m_op(m_x)
    n_x = n_op(n_x)
    check(m_x, n_x)


# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=no-self-use
@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.erf, mnm._op.sym.erf),
        (torch.nn.ReLU(), mnm._op.sym.relu),
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops_with_grad(ops, shape, dtype, ctx):
    class Unary(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return m_op(x)
    t_op, m_op = ops
    model = Unary()
    m_x, t_x = randn_torch(shape, dtype=dtype, ctx=ctx)
    m_x.requires_grad = True
    m_y = model(m_x)
    t_y = t_op(t_x)
    # check forward
    check_torch(m_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x.grad, t_x.grad)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("ops", [
    (np.log, mnm.log),
    (np.sqrt, mnm.sqrt),
])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops_pos(ops, shape, dtype, ctx):
    n_op, m_op = ops
    m_x, n_x = randn_pos(shape, dtype=dtype, ctx=ctx)
    m_x = m_op(m_x)
    n_x = n_op(n_x)
    check(m_x, n_x)


@pytest.mark.parametrize("ctx", get_ctx_list())
def test_shape(ctx):
    shape = (3, 6, 9)
    m_x = mnm.array(np.random.randn(*shape).astype('float32'), ctx=ctx)
    m_shape = mnm.shape(m_x)
    assert tuple(m_shape) == shape


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(1, 2, 3, 4), (4, 3, 2, 1), (2, 4, 1, 3)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cache(ctx, shape, dtype):
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_y, n_y = mnm.cos(m_x), np.cos(n_x)
    check(m_y, n_y)


if __name__ == "__main__":
    pytest.main([__file__])
