from numbers import Number

import numpy as np

from mnm._core.ndarray import _create_by_pair, ndarray
from mnm._core.value import (BoolValue, FloatValue, IntValue, StringValue,
                             TensorValue, Value, BoundExpr)


def ToAny(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if a is None:
        return None

    if isinstance(a, (Number, str)):
        return a

    return ToTensor(a)


def ToTensor(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if not isinstance(a, np.ndarray):
        a = np.array(a)
    # TODO(@junrushao1994): save this FFI call

    return Value.as_const_expr(TensorValue.from_numpy(a))


def ToIntTuple(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if isinstance(a, np.ndarray):
        a = a.tolist()

    if isinstance(a, Number):
        if int(a) != a:
            raise ValueError("Cannot convert to List[int]")

        return int(a)

    if not isinstance(a, (tuple, list)):
        raise ValueError("Cannot convert to List[int]")
    result = []

    for item in a:
        if isinstance(item, Number) and int(item) == item:
            result.append(int(item))
        else:
            raise ValueError("Cannot convert to List[int]")

    return result


def ToOptionalIntTuple(a):
    return None if a is None else ToIntTuple(a)


def ToInt(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and int(a) == a:
        return int(a)
    raise ValueError("Cannot convert to int")


def ToDouble(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and float(a) == a:
        return float(a)
    raise ValueError("Cannot convert to double")


def ToBool(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()

    if isinstance(a, Number) and bool(a) == a:
        return bool(a)
    raise ValueError("Cannot convert to bool")


def ToString(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle._expr

    if isinstance(a, str):
        return a
    raise ValueError("Cannot convert to str")


def Ret(a):
    if isinstance(a, (IntValue, FloatValue, StringValue)):
        return a.data
    if isinstance(a, BoolValue):
        return bool(a.data)
    if isinstance(a, BoundExpr):
        return ndarray(a)
    if isinstance(a, tuple):
        return tuple(map(Ret, a))
    if isinstance(a, list):
        return list(map(Ret, a))
    raise NotImplementedError(type(a))
