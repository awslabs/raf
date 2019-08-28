from .._core.bound_expr import BoundExpr
from .._core.executor import Interpreter
from .._core.ndarray import _create_by_pair
from .._core.op import get_op
from .._ffi._tvm import relay


def create_op(op_name: str, eager: bool = True):
    op = get_op(op_name)

    def body(args, attrs):
        args = tuple(arg._expr for arg in args)
        expr = relay.Call(op=op, args=args, attrs=attrs)
        if not eager:
            return _create_by_pair(expr=expr, value=None)
        return Interpreter.GLOBAL(expr)

    return body
