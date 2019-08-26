from ._tvm import _init_api


def CreateInterpreter(module) -> callable:
    pass


_init_api("mnm.executor", "mnm._ffi.executor")
