import numpy as np

from .base import set_module
from .bound_expr import BoundExpr
from .context import Context


@set_module("mnm")
class ndarray(object):

    def __init__(self, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        a = np.ndarray(shape=shape, dtype=dtype, buffer=buffer,
                       offset=offset, strides=strides, order=order)
        self.__handle = BoundExpr.const(a)

    @property
    def ctx(self):
        ctx = self.__handle._value.contents.ctx
        return Context.create(ctx.device_type, ctx.device_id)

    @property
    def ndim(self):
        return int(self.__handle._value.contents.ndim)

    @property
    def dtype(self):
        return str(self.__handle._value.contents.dtype)

    @property
    def shape(self):
        return tuple(self.__handle._value.contents.shape[i] for i in range(self.ndim))

    @property
    def strides(self):
        # TODO(@junrushao1994): check if they are in `numel` or `bytes`
        return tuple(self.__handle._value.contents.strides[i] for i in range(self.ndim))
