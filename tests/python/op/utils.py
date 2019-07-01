import sys
import mnm
from mnm._ffi.op import MakeOutput
from mnm._ffi._tvm import make_node


def invoke_make_output(op_name, attrs_name, args, **attrs):
    if not isinstance(args, tuple):
        args = (args, )
    if attrs_name:
        attrs = make_node(attrs_name, **attrs)
    else:
        attrs = None
    return mnm._ffi.op.MakeOutput(op_name, args, attrs)


if __name__ == "__main__":
    invoke_make_output("mnm.op.conv2d", "", None)
