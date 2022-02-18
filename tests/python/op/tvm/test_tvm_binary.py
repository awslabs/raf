# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=protected-access,attribute-defined-outside-init
# pylint: disable=no-member, no-self-use, too-many-locals, too-many-arguments
import numpy as np
import pytest
import torch
import mnm
from mnm.testing import get_testable_devices, randn, randn_torch, check, run_vm_model, with_seed


class BinaryModel(mnm.Model):
    def build(self, op):
        self.op = op

    @mnm.model.trace
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
        (np.maximum, mnm._op.sym.maximum),
        (np.greater, mnm._op.sym.greater),
        (np.minimum, mnm._op.sym.minimum),
        (np.floor_divide, mnm._op.sym.floor_divide),
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
        (torch.mul, mnm._op.sym.multiply),
        (torch.div, mnm._op.sym.divide),
        (torch.pow, mnm._op.sym.power),
        (torch.add, mnm._op.sym.add),
        (torch.sub, mnm._op.sym.subtract),
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
        (np.logical_and, mnm._op.sym.logical_and),
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
    "ops", [(np.right_shift, mnm._op.sym.right_shift), (np.left_shift, mnm._op.sym.left_shift)]
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
        (torch.eq, mnm._op.sym.equal),
        (torch.ne, mnm._op.sym.not_equal),
        (torch.lt, mnm._op.sym.less),
        (torch.le, mnm._op.sym.less_equal),
        (torch.gt, mnm._op.sym.greater),
        (torch.ge, mnm._op.sym.greater_equal),
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
