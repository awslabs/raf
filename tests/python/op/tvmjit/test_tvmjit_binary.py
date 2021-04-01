# pylint: disable=protected-access,attribute-defined-outside-init
import numpy as np
import pytest
import torch
import mnm
from mnm.testing import get_device_list, randn, randn_torch, check, run_vm_model


class BinaryModel(mnm.Model):
    def build(self, op):
        self.op = op

    @mnm.model.trace
    def forward(self, x1, x2):
        return self.op(x1, x2)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (np.add, mnm._op.sym.add),
    (np.subtract, mnm._op.sym.subtract),
    (np.maximum, mnm._op.sym.maximum),
    (np.greater, mnm._op.sym.greater),
    (np.minimum, mnm._op.sym.minimum),
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary_ops(ops, shape, dtype, device):
    n_op, m_op = ops
    model = BinaryModel(m_op)
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device)
    m_y = model(m_x1, m_x2)
    v_y = run_vm_model(model, device, [m_x1, m_x2])
    n_y = n_op(n_x1, n_x2)
    check(m_y, n_y)
    check(v_y, n_y)


# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=protected-access
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (torch.mul, mnm._op.sym.multiply),
    (torch.div, mnm._op.sym.divide),
    (torch.pow, mnm._op.sym.power),
    (torch.add, mnm._op.sym.add),
    (torch.sub, mnm._op.sym.subtract),
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary_ops_with_grad(ops, shape, dtype, device):
    t_op, m_op = ops
    m_x1, t_x1 = randn_torch(shape[0], dtype=dtype, device=device, requires_grad=True)
    m_x2, t_x2 = randn_torch(shape[1], dtype=dtype, device=device, requires_grad=True)
    model = BinaryModel(m_op)
    # check forward
    m_y = model(m_x1, m_x2)
    v_y = run_vm_model(model, device, [m_x1, m_x2])
    t_y = t_op(t_x1, t_x2)
    check(m_y, t_y)
    check(v_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(m_y.shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x1.grad, t_x1.grad)
    check(m_x2.grad, t_x2.grad)


#logical_and only allows bool input s
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (np.logical_and, mnm._op.sym.logical_and),
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["bool"])
def test_binary_bool_ops(ops, shape, dtype, device):
    n_op, m_op = ops
    model = BinaryModel(m_op)
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device)
    m_y = model(m_x1, m_x2)
    v_y = run_vm_model(model, device, [m_x1, m_x2])
    n_y = n_op(n_x1, n_x2)
    check(m_y, n_y)
    check(v_y, n_y)


# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=protected-access
# pylint: disable=no-self-use
#pylint: disable=too-many-locals
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("ops", [
    (np.right_shift, mnm._op.sym.right_shift),
    (np.left_shift, mnm._op.sym.left_shift)
])
@pytest.mark.parametrize("shape", [
    [(), (1, 2)],
    [(1, 2), (2, 1)],
    [(3, 3), (1, 1)]
])
@pytest.mark.parametrize("dtype", ["uint16", "uint8", "uint32"])
def test_int_ops_with_grad(ops, shape, dtype, device):
    n_op, m_op = ops
    model = BinaryModel(m_op)
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device)
    m_x1.requires_grad = True
    m_x2.requires_grad = True

    # check forward
    m_y = model(m_x1, m_x2)
    v_y = run_vm_model(model, device, [m_x1, m_x2])
    n_y = n_op(n_x1, n_x2)
    check(m_y, n_y)
    check(v_y, n_y)
    # check backward
    m_dy = randn(m_y.shape, dtype=dtype, device=device)[0]
    m_y.backward(m_dy)
    check(m_x1.grad, 0.)

if __name__ == "__main__":
    pytest.main([__file__])
