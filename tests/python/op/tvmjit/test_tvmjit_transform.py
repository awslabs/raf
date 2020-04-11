from functools import reduce
import operator

import numpy as np
import pytest
import topi.testing

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


def randint(shape, *, low=0, high=None, ctx="cpu", dtype="int64"):
    x = np.random.randint(low, high, shape)
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
@pytest.mark.parametrize("shape", [
    [(5, 4, 3), (1, 2)],
    [(6, 5), (2, 2)],
    [(1, 1), (2, 2, 2)],
])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_take(shape, axis, ctx):
    size = reduce(operator.mul, shape[0], 1) if axis is None else shape[0][axis]
    m_x, n_x = randn(shape[0], ctx=ctx)
    m_indices, n_indices = randint(shape[1], low=0, high=size, ctx=ctx)
    m_y = mnm.take(m_x, m_indices, axis=axis)
    n_y = np.take(n_x, n_indices, axis=axis, mode="clip")
    check(m_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("max_length", [3, 4, 5, 6])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
@pytest.mark.parametrize("other_feature_dims", [[1, 2], [3, 4], [5, 6]])
@pytest.mark.parametrize("axis", [0, 1])
def test_sequence_mask(max_length, batch_size, other_feature_dims,
                       axis, ctx):
    x_shape = [max_length, batch_size] if axis == 0 else [batch_size, max_length]
    x_shape += other_feature_dims
    m_x, n_x = randn(x_shape, ctx=ctx)
    m_length, n_length = randint([batch_size], low=0, high=max_length, ctx=ctx)
    m_y = mnm.sequence_mask(m_x, m_length, axis=axis, mask_value=-10)
    n_y = topi.testing.sequence_mask(n_x, n_length, axis=axis, mask_value=-10)
    check(m_y, n_y)


if __name__ == "__main__":
    pytest.main([__file__])
