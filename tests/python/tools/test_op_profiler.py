# pylint: disable=no-self-use,protected-access
import pytest

import mnm
from mnm._ffi.op_profiler import Profile, ResetCache, GetCacheSize
from mnm.testing import get_testable_devices, run_infer_type, randn


@pytest.mark.parametrize("device", get_testable_devices())
def test_single_op(device):
    data = mnm.ir.var("x", shape=(16, 16))
    expr = mnm.ir.op.softmax(data)
    expr = run_infer_type(expr).body
    lat = Profile(expr, mnm.Device(device))
    assert lat > 0


def test_no_compute_op():
    data = mnm.ir.var("x", shape=(16, 16))
    expr = run_infer_type(data)  # expr is a function node so no way to profile it.
    lat = Profile(expr, mnm.Device("cpu"))
    assert lat == 0


@pytest.mark.parametrize("device", get_testable_devices())
def test_closure(device):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data):
            out = mnm.relu(data)
            out = mnm.log(out)
            return out

    model = Model()
    m_x, _ = randn((10, 20), device=device)
    mod = model._internal(m_x).mod
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    mod = mnm._ffi.pass_.FuseTVM()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)

    lat = Profile(mod["main"].body, mnm.Device(device))
    assert lat > 0


def test_cache():
    device = mnm.Device("cpu")
    data = mnm.ir.var("x", shape=(16, 16))
    expr = mnm.ir.op.softmax(data)
    expr = run_infer_type(expr).body

    ResetCache(device)
    assert GetCacheSize(device) == 0
    Profile(expr, device)
    assert GetCacheSize(device) == 1


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_workspace():
    data = mnm.ir.var("x", shape=(2, 3, 14, 14))
    weight = mnm.ir.var("w", shape=(3, 3, 3, 3))
    expr = mnm.ir.op.conv2d(data, weight)
    expr = run_infer_type(expr).body
    lat = Profile(expr, mnm.Device("cuda"))
    assert lat > 0


if __name__ == "__main__":
    pytest.main([__file__])
