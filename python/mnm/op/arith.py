from typing import Union, Tuple
from .._core.ndarray import ndarray
from .._core.op import get_op
from .._ffi._tvm import _make_node
from ._typing import array_like, scalar
from ._util import int2tuple
from .imports import array as _array
from ._typing import _ARG_TYPE_GUARDS, _RET_TYPE_GUARDS


def add(x1: array_like,
        x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.add")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def subtract(x1: array_like,
             x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.subtract")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def multiply(x1: array_like,
             x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.multiply")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def divide(x1: array_like,
           x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.divide")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def mod(x1: array_like,
        x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.mod")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def less(x1: array_like,
         x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.less")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def less_equal(x1: array_like,
               x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.less_equal")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def greater(x1: array_like,
            x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.greater")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def greater_equal(x1: array_like,
                  x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.greater_equal")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def equal(x1: array_like,
          x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.equal")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def not_equal(x1: array_like,
              x2: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x1 = _ARG_TYPE_GUARDS[array_like](x1, "x1")
    x2 = _ARG_TYPE_GUARDS[array_like](x2, "x2")
    f = get_op("mnm.op.not_equal")
    res = f(eager=True,
            args=[x1, x2],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def negative(x: array_like) -> Union[ndarray, scalar]:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[array_like](x, "x")
    f = get_op("mnm.op.negative")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[Union[ndarray, scalar]](res, "return value")
    return res

def copy(x: ndarray) -> ndarray:
    """ Reserved for doc string... """
    x = _ARG_TYPE_GUARDS[ndarray](x, "x")
    f = get_op("mnm.op.copy")
    res = f(eager=True,
            args=[x],
            attrs=None)
    res = _RET_TYPE_GUARDS[ndarray](res, "return value")
    return res

