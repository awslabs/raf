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

# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
import pytest
import mnm
from mnm.testing import randn
from mnm.testing.schedule_verifier import verify_schedule


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_simple_branches():
    class Model(mnm.Model):
        """
         ┌───────x──────┐
         │       │      │
         ▼       ▼      ▼
        atan    atan   atan
         │       │      │
         │       ▼      ▼
         │      atan   atan
         │       │      │
         │       │      ▼
         └───┐   │     atan
             │   │   ┌───
             ▼   ▼   ▼
            concatenate
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)

            p_1 = mnm.atan(x)
            p_1 = mnm.atan(p_1)

            p_2 = mnm.atan(x)
            p_2 = mnm.atan(p_2)
            p_2 = mnm.atan(p_2)
            return mnm.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(
        config={
            "mnm.stream_schedule.ios.block_max_size": 20,
            "mnm.stream_schedule.ios.max_stream_num": 8,
            "mnm.stream_schedule.ios.max_stage_ops": 20,
            "mnm.stream_schedule.ios.search_group_combination": False,
            "mnm.stream_schedule.ios.warmup": 1,
            "mnm.stream_schedule.ios.number": 6,
            "mnm.stream_schedule.ios.repeat": 6,
            "mnm.stream_schedule.ios.verbose": True,
        }
    ):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseDialect()(mod)
        mod = mnm._ffi.pass_.FuseTVM()(mod)
        mod = mnm._ffi.pass_.DispatchDialect()(mod)
        mod = mnm._ffi.pass_.EraseType()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
        mod = mnm._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_branch_in_branch():
    class Model(mnm.Model):
        """
         ┌───────x──────┐
         │       │      │
         ▼       ▼      ▼
        atan    atan   atan
         │       │      │
         │       ▼      └─┐
         │   ┌──atan──┐   │
         │   │        │   ▼
         │   ▼        ▼  atan
         │  atan     atan │
         │   │        │   │
         │   ▼        ▼   │
         │   concatenate  │
         │       │        ▼
         └───┐   │   ┌───atan
             │   │   │
             ▼   ▼   ▼
            concatenate
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)

            p_1 = mnm.atan(x)
            p_1 = mnm.atan(p_1)
            p_1a = mnm.atan(p_1)
            p_1b = mnm.atan(p_1)
            p_1 = mnm.concatenate([p_1a, p_1b])

            p_2 = mnm.atan(x)
            p_2 = mnm.atan(p_2)
            p_2 = mnm.atan(p_2)
            return mnm.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(
        config={
            "mnm.stream_schedule.ios.block_max_size": 20,
            "mnm.stream_schedule.ios.max_stream_num": 8,
            "mnm.stream_schedule.ios.max_stage_ops": 20,
            "mnm.stream_schedule.ios.search_group_combination": False,
            "mnm.stream_schedule.ios.warmup": 1,
            "mnm.stream_schedule.ios.number": 6,
            "mnm.stream_schedule.ios.repeat": 6,
            "mnm.stream_schedule.ios.verbose": True,
        }
    ):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseDialect()(mod)
        mod = mnm._ffi.pass_.FuseTVM()(mod)
        mod = mnm._ffi.pass_.DispatchDialect()(mod)
        mod = mnm._ffi.pass_.EraseType()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
        mod = mnm._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_stacked_blocks():
    class Model(mnm.Model):
        """
         ┌────x────┐
         │    │    │
         │    │    ▼
         │    │   atan
         ▼    ▼    │
        atan atan  ▼
         │    │   atan
         │    │    │
         ▼    ▼    ▼
         concatenate
         │    │    │
         │    │    ▼
         │    │   atan
         ▼    ▼    │
        atan atan  ▼
         │    │   atan
         │    │    │
         ▼    ▼    ▼
         concatenate
        """

        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p_0 = mnm.atan(x)
            p_1 = mnm.atan(x)
            p_2 = mnm.atan(x)
            p_2 = mnm.atan(p_2)
            x = mnm.concatenate([p_0, p_1, p_2])
            p_0 = mnm.atan(x)
            p_1 = mnm.atan(x)
            p_2 = mnm.atan(x)
            p_2 = mnm.atan(p_2)
            return mnm.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with mnm.ir.PassContext(
        config={
            "mnm.stream_schedule.ios.block_max_size": 20,
            "mnm.stream_schedule.ios.max_stream_num": 8,
            "mnm.stream_schedule.ios.max_stage_ops": 20,
            "mnm.stream_schedule.ios.search_group_combination": False,
            "mnm.stream_schedule.ios.warmup": 1,
            "mnm.stream_schedule.ios.number": 6,
            "mnm.stream_schedule.ios.repeat": 6,
            "mnm.stream_schedule.ios.verbose": True,
        }
    ):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseDialect()(mod)
        mod = mnm._ffi.pass_.FuseTVM()(mod)
        mod = mnm._ffi.pass_.DispatchDialect()(mod)
        mod = mnm._ffi.pass_.EraseType()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
        mod = mnm._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
