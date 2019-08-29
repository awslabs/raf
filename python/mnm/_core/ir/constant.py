from numbers import Number

import numpy as np

from ..._ffi.ir import _make
from ..._ffi.ir.constant import ExtractValue
from ..value import BoolValue, FloatValue, IntValue, TensorValue, Value

VALUE_MAKER = {
    int: IntValue,
    float: FloatValue,
    bool: BoolValue,
    np.ndarray: TensorValue.from_numpy
}


def ConstantExpr(value):
    if isinstance(value, Value):
        return _make.Constant(value)
    maker = VALUE_MAKER.get(type(value), None)
    if maker is not None:
        return _make.Constant(maker(value))
    raise NotImplementedError
