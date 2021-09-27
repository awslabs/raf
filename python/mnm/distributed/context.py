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

    @property
    def zero_opt_level(self):
        return self.zero_opt_level_

    @zero_opt_level.setter
    def zero_opt_level(self, value):
        self.zero_opt_level_ = value
        ffi.ZeroOpt(value)

    @property
    def auto_dp_profiling_start_iter(self):
        return self.auto_dp_profiling_start_iter_

    @auto_dp_profiling_start_iter.setter
    def auto_dp_profiling_start_iter(self, value):
        self.auto_dp_profiling_start_iter_ = value
        ffi.AutoDPProfilingStartIter(value)

    @property
    def auto_dp_profiling_end_iter(self):
        return self.auto_dp_profiling_end_iter_

    @auto_dp_profiling_end_iter.setter
    def auto_dp_profiling_end_iter(self, value):
        self.auto_dp_profiling_end_iter_ = value
        ffi.AutoDPProfilingEndIter(value)


def get_context():
    return ffi.Global()
