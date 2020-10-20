# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
"""Distributed Context"""
import mnm._ffi.distributed as ffi
from mnm._core.core_utils import register_node
from mnm._ffi.distributed import _make
from mnm._lib import Object


@register_node("mnm.distributed.DistContext")
class DistContext(Object):

    def __init__(self):
        self.__init_handle_by_constructor__(_make.DistContext)

    @property
    def enable_data_parallel(self):
        return self.enable_data_parallel_

    @enable_data_parallel.setter
    def enable_data_parallel(self, value):
        self.enable_data_parallel_ = value
        ffi.EnableDataParallel(value)


def get_context():
    return ffi.Global()
