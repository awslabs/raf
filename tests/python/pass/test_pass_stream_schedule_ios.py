# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use, protected-access, unused-variable, too-many-locals, too-many-statements
import pytest
import raf
from raf.testing import randn
from raf.testing.schedule_verifier import verify_schedule


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_simple_branches():
    class Model(raf.Model):
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

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)

            p_1 = raf.atan(x)
            p_1 = raf.atan(p_1)

            p_2 = raf.atan(x)
            p_2 = raf.atan(p_2)
            p_2 = raf.atan(p_2)
            return raf.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(
        config={
            "raf.stream_schedule.ios.block_max_size": 20,
            "raf.stream_schedule.ios.max_stream_num": 8,
            "raf.stream_schedule.ios.max_stage_ops": 20,
            "raf.stream_schedule.ios.search_group_combination": False,
            "raf.stream_schedule.ios.warmup": 1,
            "raf.stream_schedule.ios.number": 6,
            "raf.stream_schedule.ios.repeat": 6,
            "raf.stream_schedule.ios.verbose": True,
        }
    ):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.DispatchDialect()(mod)
        mod = raf._ffi.pass_.EraseType()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
        mod = raf._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_branch_in_branch():
    class Model(raf.Model):
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

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)

            p_1 = raf.atan(x)
            p_1 = raf.atan(p_1)
            p_1a = raf.atan(p_1)
            p_1b = raf.atan(p_1)
            p_1 = raf.concatenate([p_1a, p_1b])

            p_2 = raf.atan(x)
            p_2 = raf.atan(p_2)
            p_2 = raf.atan(p_2)
            return raf.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(
        config={
            "raf.stream_schedule.ios.block_max_size": 20,
            "raf.stream_schedule.ios.max_stream_num": 8,
            "raf.stream_schedule.ios.max_stage_ops": 20,
            "raf.stream_schedule.ios.search_group_combination": False,
            "raf.stream_schedule.ios.warmup": 1,
            "raf.stream_schedule.ios.number": 6,
            "raf.stream_schedule.ios.repeat": 6,
            "raf.stream_schedule.ios.verbose": True,
        }
    ):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.DispatchDialect()(mod)
        mod = raf._ffi.pass_.EraseType()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
        mod = raf._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_ios_schedule_stacked_blocks():
    class Model(raf.Model):
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

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)
            p_1 = raf.atan(x)
            p_2 = raf.atan(x)
            p_2 = raf.atan(p_2)
            x = raf.concatenate([p_0, p_1, p_2])
            p_0 = raf.atan(x)
            p_1 = raf.atan(x)
            p_2 = raf.atan(x)
            p_2 = raf.atan(p_2)
            return raf.concatenate([p_0, p_1, p_2])

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    with raf.ir.PassContext(
        config={
            "raf.stream_schedule.ios.block_max_size": 20,
            "raf.stream_schedule.ios.max_stream_num": 8,
            "raf.stream_schedule.ios.max_stage_ops": 20,
            "raf.stream_schedule.ios.search_group_combination": False,
            "raf.stream_schedule.ios.warmup": 1,
            "raf.stream_schedule.ios.number": 6,
            "raf.stream_schedule.ios.repeat": 6,
            "raf.stream_schedule.ios.verbose": True,
        }
    ):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.DispatchDialect()(mod)
        mod = raf._ffi.pass_.EraseType()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
        mod = raf._ffi.pass_.IOSStreamSchedule()(mod)
    verify_schedule(mod)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
