from . import _make
from .._tvm import _init_api


def AssembleTensorValue(ctx, dtype, shape, strides, data):
    pass


def DeTuple(value):
    pass


_init_api("mnm.value", "mnm._ffi.value")
