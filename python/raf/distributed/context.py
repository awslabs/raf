# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
"""Distributed Context"""
import raf._ffi.distributed as ffi
from raf._core.core_utils import register_node
from raf._ffi.distributed import _make
from raf._lib import Object


@register_node("raf.distributed.DistContext")
class DistContext(Object):
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

    def dumps(self):
        attr_keys = [
            "enable_data_parallel",
            "zero_opt_level",
            "auto_dp_profiling_start_iter",
            "auto_dp_profiling_end_iter",
        ]
        return {attr: getattr(self, attr) for attr in attr_keys}

    def loads(self, context_dict):
        for attr in context_dict:
            setattr(self, attr, context_dict[attr])


def get_context():
    return ffi.GlobalDistContext()
