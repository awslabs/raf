from .._tvm import _NodeBase
from ..base import register_mnm_node
from . import _make, _module


@register_mnm_node("mnm.ir.Module")
class Module(_NodeBase):

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        _module.Add(self, var, func)

    def __getitem__(self, var):
        return _module.Lookup(self, var)


GLOBAL = Module()
