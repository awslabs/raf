# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from raf._core.value import BoolValue, FloatValue, IntValue, StringValue, TensorValue, TupleValue


def test_int_value():
    value = IntValue(1)
    data = value.value
    assert isinstance(data, int)
    assert data == 1


def test_float_value():
    value = FloatValue(3.1415926535897932384626)
    data = value.value
    assert isinstance(data, float)
    assert data == 3.1415926535897932384626


def test_bool_value():
    value = BoolValue(True)
    data = value.value
    assert isinstance(data, int)
    assert data == 1
    value = BoolValue(False)
    data = value.value
    assert isinstance(data, int)
    assert data == 0


def test_string_value():
    value = StringValue("hello world")
    data = value.value
    assert isinstance(data, str)
    assert data == "hello world"


def test_nested():
    d_0 = TensorValue.assemble((), "float32", "cpu")
    d_1 = TensorValue.assemble((3,), "float64", "cpu")
    d_2 = TensorValue.assemble((3, 2), "float16", "cpu")
    d_3 = TensorValue.assemble((3, 2, 6), "float32", "cpu")
    v_0 = TupleValue([d_0, d_1, d_2, d_3])
    v_1 = TupleValue([v_0, v_0])
    v_2 = TupleValue([])
    assert len(v_0) == 4
    assert len(v_1) == 2
    assert len(v_1[0]) == 4
    assert len(v_1[1]) == 4
    assert len(v_2) == 0
    assert not v_2
    assert v_0[0].same_as(d_0)
    assert v_0[1].same_as(d_1)
    assert v_0[2].same_as(d_2)
    assert v_0[3].same_as(d_3)
    assert v_1[0][0].same_as(d_0)
    assert v_1[0][1].same_as(d_1)
    assert v_1[0][2].same_as(d_2)
    assert v_1[0][3].same_as(d_3)
    assert v_1[1][0].same_as(d_0)
    assert v_1[1][1].same_as(d_1)
    assert v_1[1][2].same_as(d_2)
    assert v_1[1][3].same_as(d_3)


def test_assemble_0d():
    data = TensorValue.assemble((), "float32", "cpu")
    assert data.ndim == 0
    assert data.shape == ()
    assert data.strides == ()


def test_assemble_1d():
    data = TensorValue.assemble((3,), "float32", "cpu")
    assert data.ndim == 1
    assert data.shape == (3,)
    assert data.strides == (1,)


def test_assemble_2d():
    data = TensorValue.assemble((3, 2), "float32", "cpu")
    assert data.ndim == 2
    assert data.shape == (3, 2)
    assert data.strides == (2, 1)


def test_assemble_3d():
    data = TensorValue.assemble((3, 2, 6), "float32", "cpu")
    assert data.ndim == 3
    assert data.shape == (3, 2, 6)
    assert data.strides == (12, 6, 1)


def test_assemble_null_1d():
    data = TensorValue.assemble((0,), "float32", "cpu")
    assert data.ndim == 1
    assert data.shape == (0,)


def test_assemble_null_3d():
    data = TensorValue.assemble((3, 0, 2), "float32", "cpu")
    assert data.ndim == 3
    assert data.shape == (3, 0, 2)


def test_from_numpy():
    a = np.array([1, 2, 3])
    b = TensorValue.from_numpy(a)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)


if __name__ == "__main__":
    test_int_value()
    test_float_value()
    test_bool_value()
    test_string_value()
    test_nested()
    test_assemble_0d()
    test_assemble_1d()
    test_assemble_2d()
    test_assemble_3d()
    test_assemble_null_1d()
    test_assemble_null_3d()
    test_from_numpy()
