"""The extended ir nodes"""
from mnm._ffi.ir.variable import GetMayShare
from mnm._ffi.ir._make import Var as _Var
from mnm._lib import tvm as _tvm
from mnm._lib import relay as _relay


class ExtendedVar(_tvm.relay.expr.Var):  # pylint: disable=too-many-ancestors
    """An extended var in meta.

    Parameters
    ----------
        var : _tvm.relay.expr.Var
            Downcast var from a relay var to extended meta var, with no type check

    Note
    ----
    Make sure var is actually an extended meta var. This downcast comes with no type check.
    """
    def __init__(self, var: _tvm.relay.expr.Var):  # pylint: disable=super-init-not-called
        self.handle = var.handle
        _tvm._ffi.base._LIB.TVMObjectRetain(self.handle)

    @property
    def may_share(self):
        """Get may_share of the current var."""
        may_share = GetMayShare(self)
        return may_share


def extended_var(name_hint, type_annotation=None, shape=None, dtype="float32", may_share=None):
    """Create a new ExtendedVar

    This is a simple wrapper function that allows specify
    shape and dtype directly.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: Optional[_tvm.relay.Type, str]
        The type annotation on the variable.
        When type_annotation is a str, we will create a scalar variable.

    shape: Optional[List[_tvm.Expr]]
        The shape of the tensor type.

    dtype: str, optional
        The data type of the tensor.
    """
    var = _relay.var(name_hint, type_annotation, shape, dtype)
    return ExtendedVar(_Var(var.name_hint, var.type_annotation, may_share))
