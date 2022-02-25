# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import raf
import tvm

from raf._core.value import TensorValue
from raf._core.ndarray import ndarray
from raf._ffi.value import ToTVM


def test_requires_grad():
    a = raf.array([1, 2, 3], dtype="float32")
    assert not a.requires_grad
    bools = [False, True, True, False, True, False, False, False, True]
    for val in bools:
        a.requires_grad = val
        assert a.requires_grad == val


def test_mutation():
    a = raf.array([1, 2, 3], dtype="float32")
    a[:] = np.array([4, 5, 6], dtype="float64")
    assert a.shape == (3,)
    assert a.dtype == "float32"
    np.testing.assert_allclose(np.array([4, 5, 6], dtype="float32"), a.numpy())


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_move_device():
    a = raf.array([1, 2, 3], dtype="float32")
    a = a.to(device="cuda")
    assert a.device.startswith("cuda")
    np.testing.assert_allclose(np.array([1, 2, 3], dtype="float32"), a.numpy())


def test_bf16_ndarray():
    def np_float2np_bf16(arr):
        """Convert a numpy array of float to a numpy array
        of bf16 in uint16"""
        orig = arr.view("<u4")
        bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
        return np.right_shift(orig + bias, 16).astype("uint16")

    def np_float2tvm_bf16(arr):
        """Convert a numpy array of float to a TVM array
        of bf16"""
        nparr = np_float2np_bf16(arr)
        return tvm.nd.empty(nparr.shape, "bfloat16").copyfrom(nparr)

    def np_bf162np_float(arr):
        """Convert a numpy array of bf16 (uint16) to a numpy array
        of float"""
        u32 = np.left_shift(arr.astype("uint32"), 16)
        return u32.view("<f4")

    n_x = np.array([1, 2, 3], dtype="float32")
    t_x = np_float2tvm_bf16(n_x)
    m_x = ndarray.from_tensor_value(TensorValue.from_tvm(t_x))
    assert m_x.dtype == "bfloat16"
    n_y = ToTVM(m_x._ndarray__value)  # pylint: disable=protected-access
    n_y = tvm.nd.empty(n_y.shape, "uint16").copyfrom(n_y)
    n_y = np_bf162np_float(n_y.numpy())
    np.testing.assert_equal(n_x, n_y)


if __name__ == "__main__":
    pytest.main([__file__])
