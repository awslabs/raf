import mnm._ffi.ir.module as ffi
from mnm._core.core_utils import NodeBase, register_node
from mnm._ffi.ir import _make


@register_node("mnm.ir.Module")
class Module(NodeBase):

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        ffi.Add(self, var, func)

    def __getitem__(self, var):
        return ffi.Lookup(self, var)

def get_global():
    return ffi.Global()
