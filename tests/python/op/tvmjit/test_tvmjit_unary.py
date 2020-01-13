import numpy as np
import pytest

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
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops(ops, shape, dtype, ctx):
    n_op, m_op = ops
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_x = m_op(m_x)
    n_x = n_op(n_x)
    check(m_x, n_x)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("ops", [
    (np.log, mnm.log),
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
