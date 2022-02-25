# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import raf
from raf.testing import randn, get_testable_devices
from raf._lib import relay, tvm
from raf._core.core_utils import DEVICE_TYPE_MAP
from raf._core.device import Device
from raf._core.module import IRModule
from raf._ffi.pass_ import ContextAnalysis, FromRelay, InferType

# pylint: disable=invalid-name, no-self-use, redefined-builtin, too-many-locals, unused-variable


@pytest.mark.parametrize("dev", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_basic(dev, shape):
    # pylint: disable=protected-access
    # Create a symbolic model and run it
    class Add(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):  # pylint: disable=no-self-use
            return raf.add(x, y)

    # Get a Relay func
    model = Add()
    m_x, _ = randn(shape, device=dev)
    m_y, _ = randn(shape, device=dev)
    _ = model(m_x, m_y)
    func = model._internal().mod["main"]

    # Create a RAF module and set the func as main
    mod = IRModule.from_expr(func)
    # Propagate types.
    mod = InferType()(mod)

    # Performance context analysis
    ca = ContextAnalysis(mod, Device(dev))

    # No device info is propagated. Everything is on the default device.
    dev_type_id = DEVICE_TYPE_MAP[dev]
    assert all([d.device_type == dev_type_id for _, d in ca.items()])


def test_device_copy():
    if not raf.build.with_cuda():
        return

    x = relay.var("x", shape=(2, 3))
    y = relay.var("y", shape=(2, 3))
    x1 = relay.op.device_copy(x, tvm.cpu(), tvm.cuda())
    y1 = relay.op.device_copy(y, tvm.cpu(), tvm.cuda())
    out = x1 + y1
    func = relay.Function([x, y], out)
    mod = tvm.IRModule.from_expr(func)
    # Create a RAF module and set the func as main
    mod = FromRelay()(mod)
    mod = InferType()(mod)
    ca = ContextAnalysis(mod, Device("cpu"))

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.cuda().device_type
    for expr, dev in ca.items():
        if isinstance(expr, relay.Call):
            assert dev.device_type == gpu_dev
        elif isinstance(expr, relay.Var):
            if expr.name_hint == "x" or expr.name_hint == "y":
                assert dev.device_type == cpu_dev
            else:
                assert dev.device_type == gpu_dev
        elif isinstance(expr, relay.Constant):
            assert dev.device_type == gpu_dev


@pytest.mark.skip(reason="Enable the test when vm dialects have type inference.")
@pytest.mark.parametrize(
    "shape",
    [
        [3, 3],
    ],
)
def test_memory_alloc(shape):
    if not raf.build.with_cuda():
        return

    dev = "cuda"
    # pylint: disable=protected-access

    class Model(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = raf.add(x, x)
            z = raf.add(x, y)
            return z

    model_before = Model()
    model_before.infer_mode()
    m_x, _ = randn(shape, device=dev)
    func = model_before._internal(m_x).mod["main"]
    mod = IRModule.from_expr(func)
    mod = InferType()(mod)
    with Device(dev):
        mod = raf._ffi.pass_.ManifestAlloc(mod)
    mod = InferType()(mod)
    ContextAnalysis(mod, Device("cpu"))
    # TODO(zhiics) Check device info of different nodes.


if __name__ == "__main__":
    pytest.main([__file__])
