import numpy as np

import mnm
from mnm._core.value import FloatValue, IntValue, TensorValue, Value
from mnm._ffi.ir.constant import ExtractValue


def test_constant_int():
    const = Value.as_const_expr(IntValue(5))
    assert ExtractValue(const).value == 5
    const = mnm.ir.const(5)
    assert ExtractValue(const).value == 5


def test_constant_float():
    const = Value.as_const_expr(FloatValue(3.1415926535897932384626))
    assert ExtractValue(const).value == 3.1415926535897932384626
    const = mnm.ir.const(3.1415926535897932384626)
    assert ExtractValue(const).value == 3.1415926535897932384626


def test_constant_tensor():
    a = np.array([1, 2, 3])
    const = Value.as_const_expr(TensorValue.from_numpy(a))
    b = ExtractValue(const)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)
    const = mnm.ir.const(a)
    b = ExtractValue(const)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)


if __name__ == "__main__":
    test_constant_int()
    test_constant_float()
    test_constant_tensor()
