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

def randnbool(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randint(0, 2, size=shape)
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
        (np.argmax, mnm.argmax),
        (np.argmin, mnm.argmin),
    ])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_ops(ops, shape, dtype, axis, ctx):
    n_op, m_op = ops
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_x = m_op(m_x, axis)
    n_x = n_op(n_x, axis)
    check(m_x, n_x)

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize(
    "ops",
    [
        (np.all, mnm.all),
        (np.any, mnm.any),
    ])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [(1), (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
def test_all_any_ops(ops, shape, axis, keepdims, ctx):
    n_op, m_op = ops
    m_x, n_x = randnbool(shape, dtype=bool, ctx=ctx)
    m_x = m_op(m_x, axis=axis, keepdims=keepdims)
    n_x = n_op(n_x, axis=axis, keepdims=keepdims)
    check(m_x, n_x)


if __name__ == "__main__":
    pytest.main([__file__])
