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


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("ops", [
    (np.add, mnm.add),
    (np.subtract, mnm.subtract),
    (np.multiply, mnm.multiply),
    (np.maximum,  mnm.maximum),
    (np.minimum,  mnm.minimum),
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary_ops(ops, shape, dtype, ctx):
    n_op, m_op = ops
    m_x1, n_x1 = randn(shape[0], dtype=dtype, ctx=ctx)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, ctx=ctx)
    m_y = m_op(m_x1, m_x2)
    n_y = n_op(n_x1, n_x2)
    check(m_y, n_y)


if __name__ == "__main__":
    pytest.main([__file__])
