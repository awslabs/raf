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


@pytest.mark.parametrize("shape", [
    (3, 16, 128, 128),
    (3, 16),
])
@pytest.mark.parametrize("ctx", get_ctx_list())
def test_bias_add(shape, ctx):
    m_x, n_x = randn(shape, ctx=ctx)
    m_b, n_b = randn([shape[1]], ctx=ctx)
    m_y = mnm.bias_add(m_x, m_b, axis=1)
    if len(shape) == 4:
        n_b = n_b[np.newaxis, :, np.newaxis, np.newaxis]
    elif len(shape) == 2:
        n_b = n_b[np.newaxis, :]
    else:
        raise NotImplementedError
    n_y = n_x + n_b
    check(m_y, n_y)


if __name__ == "__main__":
    pytest.main()
