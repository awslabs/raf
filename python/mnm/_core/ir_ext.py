"""The extended ir nodes"""
from mnm._ffi.ir.variable import GetMayShare
from mnm._lib import tvm as _tvm

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
