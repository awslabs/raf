import numpy as np

from ._tvm import relay
from .base import register_mnm_node
from .value import FloatValue, IntValue, TensorValue, Value
from .ir import Constant


@register_mnm_node("mnm.value.BoundExpr")
class BoundExpr(Value):

    def __init__(self, expr, value, executor=None):
        self.__init_handle_by_constructor__(
            value._make.BoundExpr, expr, value, executor)

    @staticmethod
    def const(a):
        if isinstance(a, np.ndarray):
            value = TensorValue.from_numpy(a)
        elif isinstance(a, int):
            value = IntValue(a)
        elif isinstance(a, float):
            value = FloatValue(a)
        else:
            raise NotImplementedError(str(type(a)))
        expr = Constant(value)
        return BoundExpr(expr=expr, value=value)
