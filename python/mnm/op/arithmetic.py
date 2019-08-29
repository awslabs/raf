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

_less = create_op("mnm.op.less")
_greater = create_op("mnm.op.greater")
_less_equal = create_op("mnm.op.less_equal")
_greater_equal = create_op("mnm.op.greater_equal")
_equal = create_op("mnm.op.equal")
_not_equal = create_op("mnm.op.not_equal")


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
def negative(x1: array_like) -> Union[ndarray, scalar]:
    return _negative(args=(x1, ), attrs=None)


@type_check
def less(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _less(args=(x1, x2), attrs=None)


@type_check
def greater(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _greater(args=(x1, x2), attrs=None)


@type_check
def less_equal(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _less_equal(args=(x1, x2), attrs=None)


@type_check
def greater_equal(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _greater_equal(args=(x1, x2), attrs=None)


@type_check
def equal(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _equal(args=(x1, x2), attrs=None)


@type_check
def not_equal(x1: array_like, x2: array_like) -> Union[ndarray, scalar]:
    return _not_equal(args=(x1, x2), attrs=None)
