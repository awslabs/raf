# pylint: disable=protected-access,attribute-defined-outside-init, no-member, no-self-use
# pylint: disable=too-many-arguments
import numpy as np
import pytest
from scipy import special
import torch

import mnm
from mnm.testing import get_device_list, randn, randn_torch, run_vm_model, check


class UnaryModel(mnm.Model):
    def build(self, op):
        self.op = op

    @mnm.model.trace
    def forward(self, x):
        return self.op(x)


def verify_unify_op(m_op, m_arg, device, ref_fwd_out, m_dy=None, ref_grad=None):
    """A helper function to verify an op."""

    model = UnaryModel(m_op)

    # Check forward and VM
    m_y = model(m_arg)
    v_y = run_vm_model(model, device, [m_arg])
    check(m_y, ref_fwd_out)
    check(v_y, ref_fwd_out)

    if m_dy is None or ref_grad is None:
        return

    # Check backward if dy is provided
    m_y.backward(m_dy)
    check(m_arg.grad, ref_grad)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize(
    "ops",
    [
        (np.copy, mnm._op.sym.copy),
        (np.ceil, mnm._op.sym.ceil),
        (np.floor, mnm._op.sym.floor),
        (np.cos, mnm._op.sym.cos),
        (np.sin, mnm._op.sym.sin),
        (np.sign, mnm._op.sym.sign),
        (np.round, mnm._op.sym.round),
        (np.abs, mnm._op.sym.abs),
        (np.exp, mnm._op.sym.exp),
        (np.arctan, mnm._op.sym.atan),
        (special.erf, mnm._op.sym.erf),
        (np.negative, mnm._op.sym.negative),
        (np.cos, mnm._op.sym.cos),
        (np.zeros_like, mnm._op.sym.zeros_like),
        (np.ones_like, mnm._op.sym.ones_like),
        (np.trunc, mnm._op.sym.trunc),
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_common_unary_ops(ops, shape, dtype, device):
    n_op, m_op = ops
    m_x, n_x = randn(shape, dtype=dtype, device=device)
    n_y = n_op(n_x)

    verify_unify_op(m_op, m_x, device, n_y)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.erf, mnm._op.sym.erf),
        (torch.nn.ReLU(), mnm._op.sym.relu),
        (torch.nn.GELU(), mnm._op.sym.gelu),
        (torch.rsqrt, mnm._op.sym.rsqrt),
        (torch.cos, mnm._op.sym.cos),
        (torch.sin, mnm._op.sym.sin),
        (torch.exp, mnm._op.sym.exp),
        (torch.atan, mnm._op.sym.atan),
        (torch.trunc, mnm._op.sym.trunc),
        (torch.tanh, mnm._op.sym.tanh)
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops_with_grad(ops, shape, dtype, device):
    t_op, m_op = ops
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device=device)
    t_y = t_op(t_x)
    t_y.backward(t_dy)

    verify_unify_op(m_op, m_x, device, t_y, m_dy, t_x.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "ops",
    [
        (torch.cos, mnm._op.sym.cos),
        (torch.sin, mnm._op.sym.sin),
        (torch.exp, mnm._op.sym.exp),
        (torch.trunc, mnm._op.sym.trunc),
        (torch.nn.ReLU(), mnm._op.sym.relu)
    ])
def test_unary_fp16_ops_with_grad(ops):
    device = "cuda"
    shape = (1, 2, 3, 4)
    dtype = "float16"

    t_op, m_op = ops
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device=device)
    t_y = t_op(t_x)
    t_y.backward(t_dy)

    verify_unify_op(m_op, m_x, device, t_y, m_dy, t_x.grad)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.log2, mnm._op.sym.log2),
        (torch.log, mnm._op.sym.log),
        (torch.sqrt, mnm._op.sym.sqrt),
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_pos_unary_ops_with_grad(ops, shape, dtype, device):
    t_op, m_op = ops
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True, positive=True)
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device=device, positive=True)
    t_y = t_op(t_x)
    t_y.backward(t_dy)

    verify_unify_op(m_op, m_x, device, t_y, m_dy, t_x.grad)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (np.log, mnm._op.sym.log),
    (np.sqrt, mnm._op.sym.sqrt),
    (np.log2, mnm._op.sym.log2),
])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_pos_unary_ops_without_grad(ops, shape, dtype, device):
    # Skip float16 tests on CPU since it may not be supported and not much performance benefit.
    if dtype == "float16" and device == "cpu":
        return

    n_op, m_op = ops
    m_x, n_x = randn(shape, dtype=dtype, device=device, positive=True)
    n_y = n_op(n_x)

    verify_unify_op(m_op, m_x, device, n_y)


# TODO(@icemelon9, @yzhliu): shape op doesn't work in the trace, so cannot test in VM.
@pytest.mark.parametrize("device", get_device_list())
def test_shape(device):
    shape = (3, 6, 9)
    m_x = mnm.array(np.random.randn(*shape).astype('float32'), device=device)
    m_shape = mnm.shape(m_x)
    assert tuple(m_shape) == shape


if __name__ == "__main__":
    pytest.main([__file__])
