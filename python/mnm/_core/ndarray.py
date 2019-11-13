import mnm._ffi.value as ffi
from mnm._core.core_utils import ctx2str, set_module
from mnm._core.value import BoundExpr


@set_module("mnm")  # pylint: disable=invalid-name
class ndarray:

    def __init__(self, bound_expr):
        self.__handle = bound_expr

    @property
    def __tensor(self):
        return self.__handle._value._tensor  # pylint: disable=protected-access

    def asnumpy(self):
        return ffi.ToTVM(self.__handle._value).asnumpy()  # pylint: disable=protected-access

    @property
    def ctx(self):
        return ctx2str(self.__tensor.handle.contents.ctx)

    @property
    def ndim(self):
        return int(self.__tensor.handle.contents.ndim)

    @property
    def dtype(self):
        return str(self.__tensor.handle.contents.dtype)

    @property
    def shape(self):
        shape_handle = self.__tensor.handle.contents.shape

        return tuple(shape_handle[i] for i in range(self.ndim))

    @property
    def strides(self):
        # TODO(@junrushao1994): check if they are in `numel` or `bytes`
        strides_handle = self.__tensor.handle.contents.strides_handle

        return tuple(strides_handle[i] for i in range(self.ndim))


class Symbol:  # pylint: disable=too-few-public-methods

    __slots__ = ["_expr"]

    def __init__(self):
        self._expr = None

    @staticmethod
    def from_expr(expr):
        ret = Symbol()
        ret._expr = expr  # pylint: disable=protected-access

        return ret


def _create_by_pair(expr, value):
    return ndarray(BoundExpr(expr=expr, value=value))


def _create_by_bind(bound_expr):
    return ndarray(bound_expr)
