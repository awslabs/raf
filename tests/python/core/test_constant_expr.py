# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name
import numpy as np

import raf
from raf._core.value import FloatValue, IntValue, TensorValue, TupleValue, Value
from raf._ffi.ir.constant import ExtractValue


def test_constant_int():
    const = Value.as_const_expr(IntValue(5))
    assert ExtractValue(const).value == 5
    const = raf.ir.const(5)
    assert ExtractValue(const).value == 5


def test_constant_float():
    const = Value.as_const_expr(FloatValue(3.1415926535897932384626))
    assert ExtractValue(const).value == 3.1415926535897932384626
    const = raf.ir.const(3.1415926535897932384626)
    assert ExtractValue(const).value == 3.1415926535897932384626


def test_constant_tuple():
    const = raf.ir.const((1.5, 4))
    v = ExtractValue(const)
    assert isinstance(v, TupleValue)
    assert len(v) == 2
    assert isinstance(v[0], FloatValue)
    assert v[0].value == 1.5
    assert isinstance(v[1], IntValue)
    assert v[1].value == 4


def test_constant_tensor():
    a = np.array([1, 2, 3])
    const = Value.as_const_expr(TensorValue.from_numpy(a))
    b = ExtractValue(const)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)
    const = raf.ir.const(a)
    b = ExtractValue(const)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)


if __name__ == "__main__":
    test_constant_int()
    test_constant_float()
    test_constant_tuple()
    test_constant_tensor()
