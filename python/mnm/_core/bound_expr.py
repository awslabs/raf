from mnm._core.core_utils import register_node
from mnm._ffi.value import _make
from mnm._lib import _NodeBase as NodeBase


@register_node("mnm.value.BoundExpr")
class BoundExpr(NodeBase):

    def __init__(self, expr, value, executor=None):
        self.__init_handle_by_constructor__(
            _make.BoundExpr, expr, value, executor)
