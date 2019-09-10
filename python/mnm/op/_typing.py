import inspect
from functools import wraps, partial
from typing import Any, List, Tuple, Union

import numpy as np

from .._core.bound_expr import BoundExpr
from .._core.ir import ConstantExpr
from .._core.ndarray import ndarray as NDArray
from .._core.value import BoolValue, FloatValue, IntValue, TensorValue
from .._ffi.value import ToTVM
from .._ffi.ir.constant import ExtractValue

scalar = Union[int, float, bool]

array_like = Union[int, float, bool,
                   np.ndarray, NDArray,
                   List[Any], Tuple[Any]]


_PY2VALUE = {
     np.ndarray: lambda x: TensorValue.from_numpy(x),
     int: IntValue, float: FloatValue, bool: BoolValue,
     list: lambda x: TensorValue.from_numpy(np.array(x)),
     tuple: lambda x: TensorValue.from_numpy(np.array(x)),
}

def _make_array_like(a, name):
    if isinstance(a, NDArray):
        return a._ndarray__handle
    if type(a) not in _PY2VALUE.keys():
        raise NotImplementedError("Conversion {} from {} to array_like".format(name, a))
    value = _PY2VALUE[type(a)](a)
    return BoundExpr(expr=ConstantExpr(value), value=value)


def _make_scalar(ty, a, name):
    if isinstance(a, ty):
        return a
    raise NotImplementedError("Conversion {} from {} to int".format(name, ty.__name__))


def _ret_make_ndarray_or_scalar(a, name):
    if isinstance(a, (int, float, bool, NDArray)):
        return a
    if isinstance(a, (IntValue, FloatValue)):
        return a.data
    if isinstance(a, BoolValue):
        return bool(a.data)
    raise NotImplementedError(
        "Conversion {} from {} to ndarray or scalar".format(name, a))


def _ret_make_ndarray(a, name):
    if isinstance(a, NDArray):
        return a
    if isinstance(a, BoundExpr):
        return NDArray(a)
    raise NotImplementedError(
        "Conversion {} from {} to ndarray or scalar".format(name, a))


def _make_ndarray(a, name):
    if isinstance(a, NDArray):
        return a._ndarray__handle
    raise NotImplementedError(
        "Conversion {} to ndarray".format(name, a))


_ARG_TYPE_GUARDS = {
    array_like: _make_array_like,
    int: partial(_make_scalar, int),
    str: partial(_make_scalar, str),
    bool: partial(_make_scalar, bool),
    float: partial(_make_scalar, float),
    NDArray: _make_ndarray,
}

_RET_TYPE_GUARDS = {
    Union[NDArray, scalar]: _ret_make_ndarray_or_scalar,
    NDArray: _ret_make_ndarray,
}
