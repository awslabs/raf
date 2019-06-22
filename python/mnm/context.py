from ._ffi._tvm import _DLContext
from .base import set_module, import_to_module


@set_module("mnm")
class Context(_DLContext):
    pass


@import_to_module("mnm")
def cpu(dev_id=0):
    return Context(1, dev_id)


@import_to_module("mnm")
def gpu(dev_id=0):
    return Context(2, dev_id)
