from ..._ffi._tvm import _NodeBase
from ..._ffi.ir import _make
from ..._ffi.ir.module import Add, Lookup
from ..base import register_mnm_node


@register_mnm_node("mnm.ir.Module")
class Module(_NodeBase):

    GLOBAL = None

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        Add(self, var, func)

    def __getitem__(self, var):
        return Lookup(self, var)


Module.GLOBAL = Module()
