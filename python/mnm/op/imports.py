import numpy as np

from .._core.base import set_module
from .._core.ndarray import _create_by_pair
from .._core.ndarray import ndarray as NDArray
from .._ffi.ir._make import Constant
from ._numpy import np_to_tensor_value


@set_module("mnm")
def ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
    npa = np.ndarray(shape, dtype=dtype, buffer=buffer,
                     offset=offset, strides=strides, order=order)
    value = np_to_tensor_value(npa, ctx=None)
    return _create_by_pair(expr=Constant(value), value=value)


@set_module("mnm")
def array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, ctx=None):
    if isinstance(object, NDArray):
        raise NotImplementedError
    npa = np.array(object, dtype=dtype, copy=copy, order=order, subok=subok)
    value = np_to_tensor_value(npa, ctx=ctx)
    return _create_by_pair(expr=Constant(value), value=value)
