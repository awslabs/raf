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
    m_x.requires_grad = True
    return m_x, n_x


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_add_to(shape, ctx):
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm.add(x, x)
    model = Add()
    m_x, _ = randn(shape, ctx=ctx)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, ctx=ctx)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 2 * n_dy
    check(m_dx, n_dx)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3,],
    [4,]
])
def test_no_grad(shape, ctx):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y, z):  # pylint: disable=no-self-use
            indices = mnm.add(y, z)
            indices = mnm.subtract(indices, z)
            indices = mnm.add(indices, z)
            return mnm.take(x, indices, axis=0)

    model = Model()
    m_x, n_x = randn(shape, ctx=ctx)
    m_y = mnm.array([1,], dtype="int64", ctx=ctx)
    m_z = mnm.array([1,], dtype="int64", ctx=ctx)
    m_out = model(m_x, m_y, m_z)  # m_out = m_x[2]
    m_dout, n_dout = randn([1,], ctx=ctx)
    m_out.backward(m_dout)
    m_dx = m_x.grad
    n_dx = np.zeros_like(n_x)
    n_dx[2] = n_dout[0]
    check(m_dx, n_dx)


if __name__ == "__main__":
    pytest.main([__file__])
