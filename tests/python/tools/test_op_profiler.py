# pylint: disable=no-self-use,protected-access
import pytest

import mnm
from mnm._ffi.op_profiler import Profile, ProfileGroup, ResetCache, GetCacheSize
from mnm.testing import get_testable_devices, run_infer_type, randn


@pytest.mark.parametrize("device_str", get_testable_devices())
def test_single_op(device_str):
    data = mnm.ir.var("x", shape=(16, 16))
    expr = mnm.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    device = mnm.Device(device_str)

    ResetCache(device)
    assert GetCacheSize(device) == 0

    lat = Profile(expr, device)
    assert len(lat) == 1 and lat[0].value > 0
    assert GetCacheSize(device) == 1

    # Should hit the cahce.
    Profile(expr, device)
    assert GetCacheSize(device) == 1

    # Should miss the cache due to different config.
    lat = Profile(expr, device, 1, 1, 2)
    assert len(lat) == 2 and lat[0].value > 0
    assert GetCacheSize(device) == 2


def test_no_compute_op():
    data = mnm.ir.var("x", shape=(16, 16))
    expr = run_infer_type(data)  # expr is a var so no way to profile it.
    lat = Profile(expr, mnm.Device("cpu"))
    assert len(lat) == 1 and lat[0].value == 0.0


@pytest.mark.parametrize("device_str", get_testable_devices())
def test_closure(device_str):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data):
            out = mnm.relu(data)
            out = mnm.log(out)
            return out

    model = Model()
    m_x, _ = randn((10, 20), device=device_str)
    mod = model._internal(m_x).mod
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    mod = mnm._ffi.pass_.FuseTVM()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)

    device = mnm.Device(device_str)

    ResetCache(device)
    assert GetCacheSize(device) == 0

    lat = Profile(mod["main"].body, device, 2, 1, 2)
    assert len(lat) == 2 and lat[0].value > 0
    assert GetCacheSize(device) == 1

    # Should hit the cahce.
    Profile(mod["main"].body, device, 2, 1, 2)
    assert GetCacheSize(device) == 1


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_workspace():
    data = mnm.ir.var("x", shape=(2, 3, 14, 14))
    weight = mnm.ir.var("w", shape=(3, 3, 3, 3))
    expr = mnm.ir.op.conv2d(data, weight)
    expr = run_infer_type(expr).body
    lat = Profile(expr, mnm.Device("cuda"))
    assert len(lat) == 1 and lat[0].value > 0


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_multi_stream():
    device = mnm.Device("cuda")
    data = mnm.ir.var("x", shape=(16, 16))
    expr = mnm.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    ResetCache(device)
    assert GetCacheSize(device) == 0

    lat = ProfileGroup([expr, expr], device, [1, 2], 2, 1, 5)
    assert len(lat) == 5 and lat[0].value > 0
    assert GetCacheSize(device) == 1

    # Should hit the cache.
    ProfileGroup([expr, expr], device, [1, 2], 2, 1, 5)
    assert GetCacheSize(device) == 1

    # Should miss the cache due to difference configs.
    lat = ProfileGroup([expr, expr], device, [1, 2], 2, 1, 1)
    assert len(lat) == 1 and lat[0].value > 0
    assert GetCacheSize(device) == 2

    # Should miss the cache due to difference stream assignments.
    ProfileGroup([expr, expr], device, [3, 4])
    assert GetCacheSize(device) == 3


if __name__ == "__main__":
    pytest.main([__file__])
