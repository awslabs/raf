from .base import set_module
from .context import Context
from .bound_expr import BoundExpr
from .._ffi.value import ToTVM


@set_module("mnm")
class ndarray(object):

    def __init__(self, bound_expr):
        self.__handle = bound_expr

    def asnumpy(self):
        return ToTVM(self.__handle._value).asnumpy()

    @property
    def ctx(self):
        ctx = self.__handle._value._tensor.handle.contents.ctx
        return Context.create(ctx.device_type, ctx.device_id)

    @property
    def ndim(self):
        return int(self.__handle._value._tensor.handle.contents.ndim)

    @property
    def dtype(self):
        return str(self.__handle._value._tensor.handle.contents.dtype)

    @property
    def shape(self):
        shape_handle = self.__handle._value._tensor.handle.contents.shape
        return tuple(shape_handle[i] for i in range(self.ndim))

    @property
    def strides(self):
        # TODO(@junrushao1994): check if they are in `numel` or `bytes`
        strides_handle = self.__handle._value._tensor.handle.contents.strides_handle
        return tuple(strides_handle[i] for i in range(self.ndim))


def _create_by_pair(expr, value):
    return ndarray(BoundExpr(expr=expr, value=value))
