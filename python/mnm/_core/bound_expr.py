from .base import register_mnm_node
from .value import Value, IntValue, FloatValue, TensorValue
from .._ffi.value import _make


@register_mnm_node("mnm.value.BoundExpr")
class BoundExpr(Value):

    def __init__(self, expr, value, executor=None):
        self.__init_handle_by_constructor__(
            _make.BoundExpr, expr, value, executor)
