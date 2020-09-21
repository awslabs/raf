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
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_argsort(ctx, shape, axis, dtype):
    m_x, n_x = randn(shape, ctx=ctx)
    np_out = np.argsort(n_x, axis)
    mx_out = mnm.argsort(m_x, axis, dtype=dtype)
    check(mx_out, np_out.astype(dtype), rtol=1e-4, atol=1e-04)


if __name__ == "__main__":
    pytest.main([__file__])
