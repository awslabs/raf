from typing import Union

from .._core.ndarray import ndarray
from ._typing import array_like, scalar, type_check
from .base import create_op

_add = create_op("mnm.op.add")
_subtract = create_op("mnm.op.subtract")
_multiply = create_op("mnm.op.multiply")
_divide = create_op("mnm.op.divide")
_mod = create_op("mnm.op.mod")
_negative = create_op("mnm.op.negative")


@type_check
def add(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _add(args=(x1, x2), attrs=None)


@type_check
def subtract(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _subtract(args=(x1, x2), attrs=None)


@type_check
def multiply(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _multiply(args=(x1, x2), attrs=None)


@type_check
def divide(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _divide(args=(x1, x2), attrs=None)


@type_check
def mod(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _mod(args=(x1, x2), attrs=None)


@type_check
def negative(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _negative(args=(x1, x2), attrs=None)
