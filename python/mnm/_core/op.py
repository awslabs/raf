from .._ffi._tvm import _get_global_func, _make_node, _NodeBase, relay
from .._ffi.op import MakeOutput
from .base import register_mnm_node
from .executor import Interpreter
from functools import partial

OP_DICT = {}


@register_mnm_node("mnm.op.OpInfo")
class OpInfo(_NodeBase):
    pass


def invoke_make_output(op_name, attrs_name, args, **attrs):
    if not isinstance(args, tuple):
        args = (args, )
    if attrs_name:
        attrs = _make_node(attrs_name, **attrs)
    else:
        attrs = None
    tmp = MakeOutput(op_name, args, attrs)
    return tmp.output


def _get_op_dict():
    f_list_name = _get_global_func("relay.op._ListOpNames")
    f_get_op = _get_global_func("relay.op._GetOp")

    res = {}

    def body(op, eager, args, attrs):
        args = tuple(arg._expr for arg in args) if eager else args
        expr = relay.Call(op=op, args=args, attrs=attrs)
        return Interpreter.GLOBAL(expr) if eager else expr

    for name in map(lambda x: x.value, f_list_name()):
        if name.startswith("mnm.op."):
            res[name] = partial(body, f_get_op(name))

    return res


def get_op(op_name):
    global OP_DICT
    # first pass
    op = OP_DICT.get(op_name, None)
    if op is not None:
        return op
    # refresh
    OP_DICT = _get_op_dict()
    # second pass
    op = OP_DICT.get(op_name, None)
    if op is not None:
        return op
    # not found
    raise NotImplementedError("Operator {} is not found".format(op_name))
