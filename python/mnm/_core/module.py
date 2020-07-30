# pylint: disable=missing-class-docstring,missing-function-docstring
"""Module that consists of global variables and functions."""
import mnm._ffi.ir.module as ffi
from mnm._core.core_utils import register_node
from mnm._ffi.ir import _make
from mnm._lib import Object
from mnm._lib import relay


@register_node("mnm.ir.Module")
class Module(Object):

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        ffi.Add(self, var, func)

    def __getitem__(self, var):
        if isinstance(var, str):
            return ffi.LookupStr(self, var)
        if isinstance(var, relay.GlobalVar):
            return ffi.Lookup(self, var)
        raise NotImplementedError(f"Module lookup for type {type(var)} is not supported")

def get_global():
    return ffi.Global()
