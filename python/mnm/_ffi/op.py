from ._op import MakeOutput
from ._tvm import _NodeBase, make_node
from .base import register_mnm_node


def invoke_make_output(op_name, attrs_name, args, **attrs):
    if not isinstance(args, tuple):
        args = (args, )
    if attrs_name:
        attrs = make_node(attrs_name, **attrs)
    else:
        attrs = None
    tmp = MakeOutput(op_name, args, attrs)
    return tmp.output


@register_mnm_node("mnm.op.OpInfo")
class OpInfo(_NodeBase):
    pass
