from ._tvm import _init_api


def MakeOutput(op_name, attrs, args):
    pass


def GetOp(op_name):
    pass


_init_api("mnm.op", "mnm._ffi._op")
