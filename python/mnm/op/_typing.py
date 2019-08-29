import inspect
from functools import wraps
from typing import Any, List, Tuple, Union

import numpy as np

from .._core.bound_expr import BoundExpr
from .._core.ir import ConstantExpr
from .._core.ndarray import ndarray as NDArray
from .._core.value import BoolValue, FloatValue, IntValue, TensorValue

scalar = Union[int, float, bool]

array_like = Union[int, float, bool,
                   np.ndarray, NDArray,
                   List[Any], Tuple[Any]]


def _make_array_like(a, name):
    if isinstance(a, NDArray):
        return a
    if isinstance(a, np.ndarray):
        return TensorValue.from_numpy(a)
    if isinstance(a, int):
        value = IntValue(a)
        return BoundExpr(expr=ConstantExpr(value), value=value)
    if isinstance(a, float):
        value = FloatValue(a)
        return BoundExpr(expr=ConstantExpr(value), value=value)
    if isinstance(a, bool):
        value = BoolValue(a)
        return BoundExpr(expr=ConstantExpr(value), value=value)
    value = TensorValue.from_numpy(np.array(a))
    return BoundExpr(expr=ConstantExpr(value), value=value)


def _make_int(a, name):
    if isinstance(a, int):
        return a
    raise NotImplementedError("Conversion {} from {} to int".format(name, a))


def _make_float(a, name):
    if isinstance(a, float):
        return a
    raise NotImplementedError("Conversion {} from {} to float".format(name, a))


def _make_str(a, name):
    if isinstance(a, str):
        return a
    raise NotImplementedError("Conversion {} from {} to str".format(name, a))


def _make_ndarray_or_scalar(a, name):
    if isinstance(a, (int, float, bool)):
        return a
    if isinstance(a, IntValue):
        return a.data
    if isinstance(a, FloatValue):
        return a.data
    if isinstance(a, BoolValue):
        return bool(a.data)
    if isinstance(a, NDArray):
        return a
    raise NotImplementedError(
        "Conversion {} from {} to ndarray or scalar".format(name, a))


_TYPE_GUARDS = {
    array_like: _make_array_like,
    int: _make_int,
    float: _make_float,
    str: _make_str,
    bool: None,  # TODO
    Union[NDArray, scalar]: _make_ndarray_or_scalar,
}


def type_check(f):
    sig = inspect.signature(f)

    @wraps(f)
    def checked_f(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for name, param in bound.arguments.items():
            ann = sig.parameters[name].annotation
            if ann is inspect.Parameter.empty:
                continue
            bound.arguments[name] = _TYPE_GUARDS[ann](param, name)
        ann = sig.return_annotation
        assert ann is not inspect.Parameter.empty
        ret = f(*bound.args, **bound.kwargs)
        ret = _TYPE_GUARDS[ann](ret, "return value")
        return ret

    return checked_f
