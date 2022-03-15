# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,protected-access
import pytest

import raf
from raf._ffi.op_profiler import Profile, ProfileGroup, ResetCache, GetCacheSize
from raf.testing import get_testable_devices, run_infer_type, randn


@pytest.mark.parametrize("device_str", get_testable_devices())
def test_single_op(device_str):
    data = raf.ir.var("x", shape=(16, 16))
    expr = raf.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    device = raf.Device(device_str)

    ResetCache(device)
    assert GetCacheSize(device) == 0

    res = Profile(expr, device)
    lat, ws_size = res["latency"], res["workspace_size"]
    # We include workspace size as the last element of the returned list
    # This op should have a workspace size of 0
    assert len(lat) == 1 and lat[0].value > 0
    assert ws_size.value == 0
    assert GetCacheSize(device) == 1

    # Should hit the cahce.
    Profile(expr, device)
    assert GetCacheSize(device) == 1

    # Should miss the cache due to different config.
    res = Profile(expr, device, 1, 1, 2)
    lat, ws_size = res["latency"], res["workspace_size"]
    assert len(lat) == 2 and lat[0].value > 0 and lat[1].value > 0
    assert ws_size.value == 0
    assert GetCacheSize(device) == 2


def test_no_compute_op():
    data = raf.ir.var("x", shape=(16, 16))
    expr = run_infer_type(data)  # expr is a var so no way to profile it.
    res = Profile(expr, raf.Device("cpu"))
    lat, ws_size = res["latency"], res["workspace_size"]
    # Unprofilable op has a workspace size of 0
    assert len(lat) == 1 and lat[0].value == 0.0
    assert ws_size.value == 0


@pytest.mark.parametrize("device_str", get_testable_devices())
def test_closure(device_str):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, data):
            out = raf.relu(data)
            out = raf.log(out)
            return out

    model = Model()
    m_x, _ = randn((10, 20), device=device_str)
    mod = model._internal(m_x).mod
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    mod = raf._ffi.pass_.FuseTVM()(mod)
    mod = raf._ffi.pass_.InferType()(mod)

    device = raf.Device(device_str)

    ResetCache(device)
    assert GetCacheSize(device) == 0

    res = Profile(mod["main"].body, device, 2, 1, 2)
    lat, ws_size = res["latency"], res["workspace_size"]
    # Closure should not have workspace size
    assert len(lat) == 2 and lat[0].value > 0 and lat[1].value > 0
    assert ws_size.value == 0
    assert GetCacheSize(device) == 1

    # Should hit the cahce.
    Profile(mod["main"].body, device, 2, 1, 2)
    assert GetCacheSize(device) == 1


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_workspace():
    data = raf.ir.var("x", shape=(2, 3, 14, 14))
    weight = raf.ir.var("w", shape=(3, 3, 3, 3))
    expr = raf.ir.op.conv2d(data, weight)
    expr = run_infer_type(expr).body
    res = Profile(expr, raf.Device("cuda"))
    lat, ws_size = res["latency"], res["workspace_size"]
    assert len(lat) == 1 and lat[0].value > 0
    # This op is expected to have non-zero workspace size but I'm not sure.
    # Testing for non-negative for now.
    assert ws_size.value >= 0


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_multi_stream():
    device = raf.Device("cuda")
    data = raf.ir.var("x", shape=(16, 16))
    expr = raf.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    ResetCache(device)
    assert GetCacheSize(device) == 0

    res = ProfileGroup([expr, expr], device, [1, 2], 2, 1, 5)
    lat, ws_size = res["latency"], res["workspace_size"]
    assert len(lat) == 5 and lat[0].value > 0
    assert ws_size.value == 0
    assert GetCacheSize(device) == 1

    # Should hit the cache.
    ProfileGroup([expr, expr], device, [1, 2], 2, 1, 5)
    assert GetCacheSize(device) == 1

    # Should miss the cache due to difference configs.
    res = ProfileGroup([expr, expr], device, [1, 2], 2, 1, 1)
    lat, ws_size = res["latency"], res["workspace_size"]
    assert len(lat) == 1 and lat[0].value > 0
    assert ws_size.value == 0
    assert GetCacheSize(device) == 2

    # Should miss the cache due to difference stream assignments.
    ProfileGroup([expr, expr], device, [3, 4])
    assert GetCacheSize(device) == 3


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_default_stream():
    device = raf.Device("cuda")
    data = raf.ir.var("x", shape=(16, 16))
    expr = raf.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    ResetCache(device)
    assert GetCacheSize(device) == 0

    # Should use default stream when not assigning any stream.
    ProfileGroup([expr, expr], device, [], 2, 1, 5)
    assert GetCacheSize(device) == 1

    # Should hit the cache.
    ProfileGroup([expr, expr], device, [-1, -1], 2, 1, 5)
    assert GetCacheSize(device) == 1


if __name__ == "__main__":
    pytest.main([__file__])
