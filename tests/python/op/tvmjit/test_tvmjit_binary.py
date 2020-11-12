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
@pytest.mark.parametrize("ops", [
    (np.add, mnm.add),
    (np.subtract, mnm.subtract),
    (np.maximum, mnm.maximum),
    (np.minimum, mnm.minimum),
    (np.greater, mnm.greater),
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


# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=protected-access
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("ops", [
    (torch.mul, mnm._op.sym.multiply),
    (torch.div, mnm._op.sym.divide),
    (torch.pow, mnm._op.sym.power),
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary_ops_with_grad(ops, shape, dtype, ctx):
    class TestModel(mnm.Model):
        def build(self, op=None):
            self.op = op

        @mnm.model.trace
        def forward(self, x1, x2):
            return self.op(x1, x2)
    t_op, m_op = ops
    m_x1, t_x1 = randn_torch(shape[0], dtype=dtype, ctx=ctx)
    m_x2, t_x2 = randn_torch(shape[1], dtype=dtype, ctx=ctx)
    m_x1.requires_grad = True
    m_x2.requires_grad = True
    model = TestModel(m_op)
    # check forward
    m_y = model(m_x1, m_x2)
    t_y = t_op(t_x1, t_x2)
    check_torch(m_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(m_y.shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x1.grad, t_x1.grad)
    check_torch(m_x2.grad, t_x2.grad)


if __name__ == "__main__":
    pytest.main([__file__])
