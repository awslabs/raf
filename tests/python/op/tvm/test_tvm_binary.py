# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,attribute-defined-outside-init
# pylint: disable=no-member, no-self-use, too-many-locals, too-many-arguments
import numpy as np
import pytest
import torch
import raf
from raf.testing import get_testable_devices, randn, randn_torch, check, run_vm_model, with_seed


class BinaryModel(raf.Model):
    def build(self, op):
        self.op = op

    @raf.model.trace
    def forward(self, x1, x2):
        return self.op(x1, x2)


def verify_op(m_op, m_args, device, ref_fwd_out, m_dy=None, ref_grads=None):
    """A helper function to verify an op."""

    model = BinaryModel(m_op)

    # Check forward and VM
    m_y = model(*m_args)
    v_y = run_vm_model(model, device, m_args)
    check(m_y, ref_fwd_out)
    check(v_y, ref_fwd_out)

    if m_dy is None or ref_grads is None:
        return

    # Check backward if dy is provided
    m_y.backward(m_dy)
    for m_arg, ref_grad in zip(m_args, ref_grads):
        check(m_arg.grad, ref_grad)


@with_seed(0)
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (np.maximum, raf._op.sym.maximum),
        (np.greater, raf._op.sym.greater),
        (np.minimum, raf._op.sym.minimum),
        (np.floor_divide, raf._op.sym.floor_divide),
    ],
)
@pytest.mark.parametrize("shape", [[(), (1, 2)], [(3, 3), (1, 1)]])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_binary_ops_without_grad(ops, shape, dtype, device):
    # Skip float16 tests on CPU since it may not be supported and not much performance benefit.
    if dtype == "float16" and device == "cpu":
        return

    n_op, m_op = ops
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device)
    n_y = n_op(n_x1, n_x2)

    verify_op(m_op, [m_x1, m_x2], device, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.mul, raf._op.sym.multiply),
        (torch.div, raf._op.sym.divide),
        (torch.pow, raf._op.sym.power),
        (torch.add, raf._op.sym.add),
        (torch.sub, raf._op.sym.subtract),
    ],
)
@pytest.mark.parametrize("shape", [[(), (1, 2)], [(3, 3), (1, 1)]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_binary_ops_with_grad(ops, shape, dtype, device):
    t_op, m_op = ops
    m_x1, t_x1 = randn_torch(shape[0], dtype=dtype, device=device, requires_grad=True)
    m_x2, t_x2 = randn_torch(shape[1], dtype=dtype, device=device, requires_grad=True)
    t_y = t_op(t_x1, t_x2)
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, device=device)
    t_y.backward(t_dy)

    verify_op(m_op, [m_x1, m_x2], device, t_y, m_dy, [t_x1.grad, t_x2.grad])


# logical_and only allows bool input s
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (np.logical_and, raf._op.sym.logical_and),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [(), (1, 2)],
        [(1, 2), (2, 1)],
    ],
)
@pytest.mark.parametrize("dtype", ["bool"])
def test_binary_bool_ops(ops, shape, dtype, device):
    n_op, m_op = ops
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device)
    n_y = n_op(n_x1, n_x2)

    verify_op(m_op, [m_x1, m_x2], device, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops", [(np.right_shift, raf._op.sym.right_shift), (np.left_shift, raf._op.sym.left_shift)]
)
@pytest.mark.parametrize("shape", [[(), (1, 2)], [(3, 3), (1, 1)]])
@pytest.mark.parametrize("dtype", ["uint16", "uint8", "uint32"])
def test_shift_ops_with_grad(ops, shape, dtype, device):
    n_op, m_op = ops
    m_x1, n_x1 = randn(shape[0], dtype=dtype, device=device, requires_grad=True)
    m_x2, n_x2 = randn(shape[1], dtype=dtype, device=device, requires_grad=True)
    n_y = n_op(n_x1, n_x2)
    m_dy = randn(n_y.shape, dtype=dtype, device=device)[0]

    verify_op(m_op, [m_x1, m_x2], device, n_y, m_dy, [0.0])


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (torch.eq, raf._op.sym.equal),
        (torch.ne, raf._op.sym.not_equal),
        (torch.lt, raf._op.sym.less),
        (torch.le, raf._op.sym.less_equal),
        (torch.gt, raf._op.sym.greater),
        (torch.ge, raf._op.sym.greater_equal),
    ],
)
@pytest.mark.parametrize("shape", [[(), (1, 2)], [(3, 3), (1, 1)]])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_logic_ops(ops, shape, dtype, device):
    # Skip float16 tests on CPU since it may not be supported and not much performance benefit.
    if dtype == "float16" and device == "cpu":
        return

    t_op, m_op = ops
    m_x1, t_x1 = randn_torch(shape[0], dtype=dtype, device=device)
    m_x2, t_x2 = randn_torch(shape[1], dtype=dtype, device=device)
    t_y = t_op(t_x1, t_x2)

    verify_op(m_op, [m_x1, m_x2], device, t_y)


if __name__ == "__main__":
    pytest.main([__file__])
