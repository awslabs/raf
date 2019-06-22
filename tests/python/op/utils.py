import sys
import mnm
# from mnm._ffi.op import MakeOutput


def invoke_make_output(op_name, attrs_name, args, *attrs):
    if not isinstance(args, tuple):
        args = (args, )
    if not attrs_name:
        attrs_name = "Dummy"
    attrs_maker = "mnm._ffi.attrs._make." + attrs_name
    module_name, maker_name = attrs_maker.rsplit('.', 1)
    attrs = getattr(sys.modules[module_name], maker_name)(*attrs)
    return mnm._ffi.op.MakeOutput(op_name, args, attrs)


if __name__ == "__main__":
    invoke_make_output("mnm.op.conv2d", "", None)
