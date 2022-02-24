# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,attribute-defined-outside-init,no-self-use
import numpy as np
import pytest
import torch
import raf
from raf.testing import get_testable_devices, randn, randn_torch, check, run_vm_model


def randnbool(shape, *, device="cpu", dtype="float32"):
    x = np.random.randint(0, 2, size=shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = raf.array(n_x, device=device)
    return m_x, n_x


def axis_exclude(input_axis, shape):
    # get the excluded axis which is not in the input axis
    # e.g.: input_axis = [1, 3] with a total of 4 dimension
    #       the reverse_axis = [0,2]
    total_dim = len(shape)
    exclude_axis = []
    if isinstance(input_axis, int):
        input_axis = [input_axis]

    for i in range(total_dim):
        if i not in input_axis:
            exclude_axis.append(i)

    return exclude_axis


class ReduceModel(raf.Model):
    def build(self, op, **kwargs):
        self.op = op
        self.attrs = kwargs

    @raf.model.trace
    def forward(self, x):
        return self.op(x, **self.attrs)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (np.argmax, raf._op.sym.argmax),
        (np.argmin, raf._op.sym.argmin),
        (np.amax, raf._op.sym.max),
        (np.amin, raf._op.sym.min),
    ],
)
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_ops(ops, shape, dtype, axis, device):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis)
    m_x, n_x = randn(shape, dtype=dtype, device=device)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    n_y = n_op(n_x, axis)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (np.sum, raf._op.sym.sum),
        (np.prod, raf._op.sym.prod),
        (np.mean, raf._op.sym.mean),
    ],
)
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("axis", [1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
# pylint: disable=too-many-arguments
def test_reduce_keepdims_ops(ops, shape, dtype, axis, keepdims, device):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis, keepdims=keepdims)
    m_x, n_x = randn(shape, dtype=dtype, device=device)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    n_y = n_op(n_x, axis=axis, keepdims=keepdims)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "ops",
    [
        (np.all, raf._op.sym.all),
        (np.any, raf._op.sym.any),
    ],
)
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
# pylint: disable=too-many-arguments
def test_all_any_ops(ops, shape, axis, keepdims, device):
    n_op, m_op = ops
    model = ReduceModel(m_op, axis=axis, keepdims=keepdims)
    m_x, n_x = randnbool(shape, dtype=bool, device=device)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    n_y = n_op(n_x, axis=axis, keepdims=keepdims)
    check(m_y, n_y)
    check(v_y, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("axis", [1, (0, 1), 0])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize(
    "ops",
    [
        (torch.sum, raf._op.sym.sum),
        (torch.mean, raf._op.sym.mean),
        (torch.prod, raf._op.sym.prod),
    ],
)
# pylint: disable=too-many-arguments,too-many-locals
def test_reduce_op_with_axis_with_grad(ops, shape, dtype, axis, keepdims, exclude, device):
    if (not isinstance(axis, int)) and (len(axis) == len(shape) and exclude):
        pytest.skip("pytroch do not support when exclude is true and all axis is used")
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    t_op, m_op = ops
    model = ReduceModel(m_op, axis=axis, keepdims=keepdims, exclude=exclude)
    # check forward
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    if exclude:
        axis = axis_exclude(axis, shape)
    if (not isinstance(axis, int)) and (ops[0] == torch.prod):
        pytest.skip("torch.prod only support axis argument as int not array of int")
    t_y = t_op(t_x, axis=axis, keepdim=keepdims)  # pylint: disable=no-member
    check(m_y, t_y)
    check(v_y, t_y)
    # check backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "ops",
    [
        (torch.sum, raf._op.sym.sum),
        (torch.mean, raf._op.sym.mean),
        (torch.prod, raf._op.sym.prod),
    ],
)
def test_reduce_op_without_axis_with_grad(ops, shape, dtype, device):
    m_x, t_x = randn_torch(shape, dtype=dtype, device=device, requires_grad=True)
    t_op, m_op = ops
    model = ReduceModel(m_op)
    # check forward
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    t_y = t_op(t_x)  # pylint: disable=no-member
    check(m_y, t_y)
    check(v_y, t_y)
    # chack backward
    m_dy, t_dy = randn_torch(t_y.shape, dtype=dtype, device=device)
    m_y.backward(m_dy)
    t_y.backward(t_dy)
    check(m_x.grad, t_x.grad)


@pytest.mark.parametrize("device", get_testable_devices())
def test_l2norm(device):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf._op.sym.l2norm(x)

    model = TestModel()
    shape = [2, 3, 4]
    m_input, t_input = randn_torch(shape, device=device)

    m_output = model(m_input)
    t_output = torch.sqrt(torch.sum(t_input * t_input))
    check(m_output, t_output)
    v_output = run_vm_model(model, device, [m_input])
    check(v_output, t_output)


if __name__ == "__main__":
    pytest.main([__file__])
