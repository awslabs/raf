# pylint: disable=missing-class-docstring,missing-function-docstring
"""Module that consists of global variables and functions."""
import mnm._ffi.ir.module as ffi
from mnm._core.core_utils import register_node
from mnm._ffi.ir import _make, AsText
from mnm._lib import Object
from mnm._lib import relay


@register_node("mnm.ir.Module")
class Module(Object):

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        self.__init_handle_by_constructor__(_make.Module, functions)

    def __setitem__(self, var, func):
        if isinstance(var, str):
            ffi.Add(self, ffi.GetGlobalVar(self, var), func)
        elif isinstance(var, relay.GlobalVar):
            ffi.Add(self, var, func)
        else:
            raise NotImplementedError(
                f"Module function assigment for type {type(var)} is not supported")

    def __getitem__(self, var):
        if isinstance(var, str):
            return ffi.LookupStr(self, var)
        if isinstance(var, relay.GlobalVar):
            return ffi.Lookup(self, var)
        raise NotImplementedError(
            f"Module lookup for type {type(var)} is not supported")

    def get_global_vars(self):
        return ffi.GetGlobalVars(self)

    def __str__(self):
        mod_str = str()
        for gvar in self.get_global_vars():
            mod_str += str(gvar) + " : " + str(AsText(self[gvar])) + "\n"
        return mod_str

    @staticmethod
    def from_expr(expr, global_funcs=None):
        """Construct a module from a standalone expression.

        Parameters
        ----------
        expr: RelayExpr
            The starting expression

        global_funcs: Optional[dict]
            Map of global vars to function definitions

        Returns
        -------
        mod: Module
            A module containing the passed definitions,
            where expr is set as the entry point
            (wrapped in a function if necessary)
        """
        funcs = global_funcs if global_funcs is not None else {}
        return ffi.FromExpr(expr, funcs)


def get_global():
    return ffi.Global()
