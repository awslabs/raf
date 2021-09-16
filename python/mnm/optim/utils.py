"""Optimizer utilities."""
from mnm._lib import relay
from mnm._ffi.ir.constant import ExtractValue
from mnm._ffi.binding import LookupBoundExpr
from mnm._core.value import NoGradValue
from mnm._core.ndarray import get_symbol_handle


def has_grad(dx):
    """ Check if dx is NoGradValue """
    def simplify(x):
        if isinstance(x, relay.Var):
            return simplify(LookupBoundExpr(x))
        if isinstance(x, relay.TupleGetItem):
            tup = simplify(x.tuple_value)
            if isinstance(tup, relay.Tuple):
                return simplify(tup[x.index])
        return x

    dx = simplify(get_symbol_handle(dx))
    if isinstance(dx, relay.Constant):
        dx = ExtractValue(dx)
        return not isinstance(dx, NoGradValue)
    return True
