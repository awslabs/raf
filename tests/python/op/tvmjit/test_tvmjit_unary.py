# pylint: disable=protected-access,attribute-defined-outside-init
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
        (special.erf, mnm._op.sym.erf),  # pylint: disable=no-member
        (np.negative, mnm._op.sym.negative),
        (np.cos, mnm._op.sym.cos),
        (np.zeros_like, mnm._op.sym.zeros_like),
        (np.ones_like, mnm._op.sym.ones_like),
        (np.trunc, mnm._op.sym.trunc),
    ])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops(ops, shape, dtype, device):
    n_op, m_op = ops
    model = UnaryModel(m_op)
    m_x, n_x = randn(shape, dtype=dtype, device=device)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    n_y = n_op(n_x)
    check(m_y, n_y)
    check(v_y, n_y)


# pylint: disable=no-member
# pylint: disable=protected-access
# pylint: disable=no-self-use
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.erf, mnm._op.sym.erf),
        (torch.nn.ReLU(), mnm._op.sym.relu),
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
    model = UnaryModel(m_op)
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    t_y = t_op(t_x)
    # check forward
    check(m_y, t_y)
    check(v_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (np.log, mnm._op.sym.log),
    (np.sqrt, mnm._op.sym.sqrt),
])
@pytest.mark.parametrize("shape", [(), (1, ), (1, 2), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary_ops_pos(ops, shape, dtype, device):
    n_op, m_op = ops
    model = UnaryModel(m_op)
    m_x, n_x = randn(shape, dtype=dtype, device=device, positive=True)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    n_y = n_op(n_x)
    check(m_y, n_y)
    check(v_y, n_y)


# TODO(@icemelon9, @yzhliu): shape op doesn't work in the trace, so cannot test in VM.
@pytest.mark.parametrize("device", get_device_list())
def test_shape(device):
    shape = (3, 6, 9)
    m_x = mnm.array(np.random.randn(*shape).astype('float32'), device=device)
    m_shape = mnm.shape(m_x)
    assert tuple(m_shape) == shape


if __name__ == "__main__":
    pytest.main([__file__])
