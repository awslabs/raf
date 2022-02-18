# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
    def size(self):
        return self.size_

    @size.setter
    def size(self, value):
        self.size_ = value
        ffi.SetGlobalSize(value)

    @property
    def rank(self):
        return self.rank_

    @rank.setter
    def rank(self, value):
        self.rank_ = value
        ffi.SetGlobalRank(value)

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
