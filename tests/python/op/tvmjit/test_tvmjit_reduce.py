# pylint: disable=protected-access,attribute-defined-outside-init
import numpy as np
import pytest
import torch
import mnm
from mnm.testing import get_ctx_list, randn, randn_torch, check, run_vm_model


def randnbool(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randint(0, 2, size=shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


class ReduceModel(mnm.Model):
    def build(self, op, **kwargs):
        self.op = op
        self.attrs = kwargs

    @mnm.model.trace
    def forward(self, x):
        return self.op(x, **self.attrs)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize(
    "ops",
    [
        (np.argmax, mnm._op.sym.argmax),
        (np.argmin, mnm._op.sym.argmin),
        (np.amax, mnm._op.sym.max),
        (np.amin, mnm._op.sym.min),
    ])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_ops(ops, shape, dtype, axis, ctx):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis)
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_y = model(m_x)
    v_y = run_vm_model(model, ctx, [m_x])
    n_y = n_op(n_x, axis)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize(
    "ops",
    [
        (np.sum, mnm._op.sym.sum),
        (np.prod, mnm._op.sym.prod)
    ])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [(1), (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
# pylint: disable=too-many-arguments
def test_reduce_keepdims_ops(ops, shape, dtype, axis, keepdims, ctx):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis, keepdims=keepdims)
    m_x, n_x = randn(shape, dtype=dtype, ctx=ctx)
    m_y = model(m_x)
    v_y = run_vm_model(model, ctx, [m_x])
    n_y = n_op(n_x, axis=axis, keepdims=keepdims)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize(
    "ops",
    [
        (np.all, mnm._op.sym.all),
        (np.any, mnm._op.sym.any),
    ])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [(1), (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
def test_all_any_ops(ops, shape, axis, keepdims, ctx):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis, keepdims=keepdims)
    m_x, n_x = randnbool(shape, dtype=bool, ctx=ctx)
    m_y = model(m_x)
    v_y = run_vm_model(model, ctx, [m_x])
    n_y = n_op(n_x, axis=axis, keepdims=keepdims)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [(1), (0, 1)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_op_with_axis(shape, dtype, axis, keepdims, ctx):
    m_x, t_x = randn_torch(shape, dtype=dtype, ctx=ctx, requires_grad=True)
    model = ReduceModel(mnm._op.sym.mean, axis=axis, keepdims=keepdims)
    # check forward
    m_y = model(m_x)
    v_y = run_vm_model(model, ctx, [m_x])
    t_y = torch.mean(t_x, axis=axis, keepdim=keepdims) # pylint: disable=no-member
    check(m_y, t_y)
    check(v_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mean_op_without_axis(shape, dtype, ctx):
    m_x, t_x = randn_torch(shape, dtype=dtype, ctx=ctx, requires_grad=True)
    model = ReduceModel(mnm._op.sym.mean)
    # check forward
    m_y = model(m_x)
    v_y = run_vm_model(model, ctx, [m_x])
    t_y = torch.mean(t_x) # pylint: disable=no-member
    check(m_y, t_y)
    check(v_y, t_y)
    # chack backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, ctx=ctx)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
