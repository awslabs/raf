from ._base import set_module
from ._ffi.context import Context


@set_module("mnm")
def cpu(dev_id=0):
    return Context.create(1, dev_id)


@set_module("mnm")
def gpu(dev_id=0):
    return Context.create(2, dev_id)
