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

def randnbool(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randint(0, 2, size=shape)
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
        (np.argmax, mnm.argmax),
        (np.argmin, mnm.argmin),
        (np.amax, mnm.max),
        (np.amin, mnm.min),
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

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [(1), (0, 1)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_op_withaxis(shape, dtype, axis, keepdims, ctx):
    class Mean(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x): # pylint: disable=no-self-use
            return mnm.mean(x, axis, keepdims)

    m_x, t_x = randn_torch(shape, dtype=dtype, ctx=ctx)
    m_x.requires_grad = True
    model = Mean()
    # check forward
    m_y = model(m_x)
    t_y = torch.mean(t_x, axis, keepdim=keepdims) # pylint: disable=no-member
    check_torch(m_y, t_y)
    # chack backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x.grad, t_x.grad)

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mean_op_withoutaxis(shape, dtype, ctx):
    class Mean(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x): # pylint: disable=no-self-use
            return mnm.mean(x)

    m_x, t_x = randn_torch(shape, dtype=dtype, ctx=ctx)
    m_x.requires_grad = True
    model = Mean()
    # check forward
    m_y = model(m_x)
    t_y = torch.mean(t_x) # pylint: disable=no-member
    check_torch(m_y, t_y)
    # chack backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check_torch(m_x.grad, t_x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
