from ._ffi._tvm import _NodeBase
from ._ffi._make import Module as make_module
from ._ffi.module import Module_Add, Module_Lookup
from .base import register_mnm_node, set_module


@set_module("mnm")
@register_mnm_node("mnm.Module")
class Module(_NodeBase):

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(make_module, functions)

    def __setitem__(self, var, func):
        Module_Add(self, var, func)

    def __getitem__(self, var):
        Module_Lookup(self, var)
