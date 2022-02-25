# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import raf
from raf._op import sym
from raf.testing import check_type, run_infer_type, randn_torch
from tvm.relay import TensorType, FuncType


# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
@pytest.mark.parametrize(
    "op",
    [
        (sym.argmax, torch.argmax, False),
        (sym.argmin, torch.argmin, False),
        (sym.max, torch.max, True),
        (sym.min, torch.min, True),
        (sym.prod, torch.prod, True),
        (sym.sum, torch.sum, True),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce(op, shape, keepdims, dtype):
    raf_fwd, torch_fwd, same = op

    axis = int(np.random.randint(-len(shape), len(shape), ()))

    class Reduce(raf.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis, keepdims=keepdims)

    model = Reduce()

    # forward
    m_x, t_x = randn_torch(shape, dtype=dtype)

    m_func = model._internal(m_x).mod["main"]
    m_func = run_infer_type(m_func)

    t_y = torch_fwd(t_x, dim=axis, keepdim=keepdims)
    x_ty = TensorType(t_x.shape, dtype=dtype)

    if torch_fwd in [torch.max, torch.min]:
        ty_shape = t_y.values.shape
    else:
        ty_shape = t_y.shape
    if not same:
        y_ty = TensorType(ty_shape, dtype="int32")
    else:
        y_ty = TensorType(ty_shape, dtype=dtype)
    desired_type = FuncType([x_ty], y_ty)

    check_type(m_func, desired_type)


# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
@pytest.mark.parametrize(
    "op",
    [
        (sym.mean, torch.mean),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_with_backward(op, shape, keepdims, dtype):
    raf_fwd, torch_fwd = op

    axis = int(np.random.randint(-len(shape), len(shape), ()))

    class Reduce(raf.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis, keepdims=keepdims)

    model = Reduce()

    # forward
    m_x, t_x = randn_torch(shape, dtype=dtype)

    m_record = model._internal(m_x)
    m_func = m_record.mod["main"]
    m_func = run_infer_type(m_func)

    # backward
    m_record.mod["main"] = m_func
    m_mod = raf._ffi.pass_.AutoDiff([])(m_record.mod)
    run_infer_type(m_mod)

    t_y = torch_fwd(t_x, dim=axis, keepdim=keepdims)
    x_ty = TensorType(t_x.shape, dtype=dtype)

    ty_shape = t_y.shape
    y_ty = TensorType(ty_shape, dtype=dtype)
    desired_type = FuncType([x_ty], y_ty)

    check_type(m_func, desired_type)


# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
@pytest.mark.parametrize(
    "op",
    [
        (sym.argmax, torch.all, False),
        (sym.argmin, torch.any, False),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [3, 2, 5, 8, 4, 7],
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "bool"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_all_any(op, shape, keepdims, dtype):
    raf_fwd, torch_fwd, same = op

    axis = int(np.random.randint(-len(shape), len(shape), ()))

    class Reduce(raf.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis, keepdims=keepdims)

    model = Reduce()

    # forward
    m_x, t_x = randn_torch(shape, dtype=dtype, requires_grad=False)

    m_func = model._internal(m_x).mod["main"]
    m_func = run_infer_type(m_func)

    t_y = torch_fwd(t_x, dim=axis, keepdim=keepdims)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    if not same:
        y_ty = TensorType(t_y.shape, dtype="int32")
    else:
        y_ty = TensorType(t_y.shape, dtype=dtype)
    desired_type = FuncType([x_ty], y_ty)

    check_type(m_func, desired_type)


if __name__ == "__main__":
    pytest.main([__file__])
