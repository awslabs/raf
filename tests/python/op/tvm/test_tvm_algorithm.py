# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,wrong-import-order,protected-access
import numpy as np
import random
import pytest
import torch
import raf
import mxnet as mx
from raf.testing import get_testable_devices, randn, randn_torch, check, run_vm_model, to_torch_dev


class TestModel(raf.Model):
    def build(self, op, **kwargs):
        self.op = op  # pylint: disable=attribute-defined-outside-init
        self.attrs = kwargs  # pylint: disable=attribute-defined-outside-init

    @raf.model.trace
    def forward(self, *args):
        return self.op(*args, **self.attrs)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 4),
        (1, 4, 6),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_argsort(device, shape, axis, dtype):
    m_x, n_x = randn(shape, device=device)
    model = TestModel(raf._op.sym.argsort, axis=axis, dtype=dtype)
    m_out = model(m_x)
    v_out = run_vm_model(model, device, [m_x])
    np_out = np.argsort(n_x, axis).astype(dtype)
    check(m_out, np_out)
    check(v_out, np_out)


# pylint: disable=too-many-locals
# pylint: disable=no-member
# pylint: disable=consider-using-in
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 4),
        (1, 4, 6),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_sort(device, shape, axis, dtype):
    m_x, n_x = randn(shape, device=device, dtype=dtype)
    m_x.requires_grad = True
    model = TestModel(raf._op.sym.sort, axis=axis)
    m_out = model(m_x)
    v_out = run_vm_model(model, device, [m_x])
    np_out = np.sort(n_x, axis)
    check(m_out, np_out)
    check(v_out, np_out)
    if dtype == "float32" or dtype == "float64":
        m_dy, n_dy = randn(m_out.shape, device=device, dtype=dtype)
        m_out.backward(m_dy)

        # ground truth
        mx_x = mx.nd.array(n_x)
        mx_x.attach_grad()
        mx_dy = mx.nd.array(n_dy)
        with mx.autograd.record():
            mx_y = mx.nd.sort(mx_x, axis)
            mx_y.backward(mx_dy)
        check(mx_x.grad, m_x.grad)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("ret_type", ["values", "both", "indices"])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("shape", [(5, 5, 5, 5, 5, 5, 5), (224, 224, 3)])
def test_topk(shape, k, axis, ret_type, is_ascend, dtype, device):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    size = 1
    for i in shape:
        size *= i
    x = np.arange(size)
    random.shuffle(x)
    x = x.reshape(shape)
    m_x = raf.array(x, dtype=dtype, device=device)
    n_x = mx.nd.array(x, dtype=dtype)

    model = TestModel(
        raf._op.sym.topk,
        k=k,
        axis=axis,
        ret_type=ret_type,
        is_ascend=is_ascend,
        dtype=dtype,
    )
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    # check forward
    if ret_type == "values":
        n_y = mx.nd.topk(n_x, k=k, axis=axis, ret_typ="value", is_ascend=is_ascend, dtype=dtype)
        check(m_y, n_y)
        check(v_y, n_y)
    else:
        n_y = mx.nd.topk(n_x, k=k, axis=axis, ret_typ=ret_type, is_ascend=is_ascend, dtype=dtype)
        if ret_type == "both":
            check(m_y[0], n_y[0])
            check(v_y[0], n_y[0])
            check(m_y[1], n_y[1])
            check(v_y[1], n_y[1])
        else:
            check(m_y, n_y)
            check(v_y, n_y)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("axis", [2, -1])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("ret_type", ["both"])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("shape", [(5, 3, 3), (5, 5, 5, 5, 5, 5)])
def test_topk_dx(shape, k, axis, ret_type, is_ascend, dtype, device):  # pylint: disable=R0915
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    if dtype == "float16" and device == "cpu":
        pytest.skip("""float16 doesn't support in cpu""")

    # Generate a set of non-duplicated numbers
    size = 1
    for i in shape:
        size *= i
    n_x = np.arange(size)
    np.random.shuffle(n_x)
    n_x = n_x.reshape(shape).astype(dtype)

    if dtype == "float16" and size >= 2048:
        pytest.skip("""For float16, shape is too big to produce non-duplicated array""")

    m_x = raf.array(n_x, dtype=dtype, device=device)
    m_x.requires_grad = True
    t_x = torch.tensor(n_x, requires_grad=True, device=to_torch_dev(device))

    model = TestModel(
        raf._op.sym.topk, k=k, axis=axis, ret_type=ret_type, is_ascend=is_ascend, dtype=dtype
    )
    m_y = model(m_x)
    m_dy, t_dy = randn_torch(m_y[0].shape, dtype=dtype, device=device, requires_grad=True)
    t_y = torch.topk(t_x, k=k, dim=axis, largest=not is_ascend)

    check(t_y[0], m_y[0])
    check(t_y[1], m_y[1])

    t_y[0].backward(t_dy)
    m_y[0].backward(m_dy)
    check(t_x.grad, m_x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
