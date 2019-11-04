import numpy as np

from mnm._ffi.ir.constant import ExtractValue
from mnm._core.ir import ConstantExpr
from mnm._core.value import FloatValue, IntValue, TensorValue


def test_constant_int():
    const = ConstantExpr(IntValue(5))
    assert ExtractValue(const).data == 5


def test_constant_float():
    const = ConstantExpr(FloatValue(3.1415926535897932384626))
    assert ExtractValue(const).data == 3.1415926535897932384626


def test_constant_tensor():
    a = np.array([1, 2, 3])
    const = ConstantExpr(TensorValue.from_numpy(a))
    b = ExtractValue(const)
    assert a.shape == b.shape
    assert a.strides == tuple(x * a.itemsize for x in b.strides)
    assert str(a.dtype) == str(b.dtype)


if __name__ == "__main__":
    test_constant_int()
    test_constant_float()
    test_constant_tensor()
