# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import raf
import tvm
from raf.testing import check
from raf._core.device import Device
from raf._core.executor import VMExecutor
from raf._core.vm_debug import VMDebugExecutor
from raf._ffi.memory_pool import InitPool
from raf.testing import get_testable_devices, randn, with_seed


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
@with_seed(0)
def test_vm_debugger(device, shape):
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

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    mod = model._internal(m_x).mod
    # disable fusion
    with raf.ir.PassContext(opt_level=1):
        executor = VMDebugExecutor(mod, device)

    # Testing whether we can get the correct intermediate tensor
    m_z = executor.make_executor()(m_x).numpy()
    ref_x = m_x.numpy()
    ref_y = ref_x + ref_x
    ref_z = model(m_x).numpy()
    check(m_z, ref_z, rtol=1e-5, atol=1e-5)
    _, ins, outs = executor.get_interm_tensors()
    check(ins[0][0], ref_x)
    check(ins[0][1], ref_x)
    check(ins[1][0], ref_x)
    check(ins[1][1], ref_y)
    check(outs[0], ref_y)
    check(outs[1], ref_z)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("pool_name", ["no_pool", "page_unit_pool"])
def test_vm_memory_profiler(device, pool_name):
    # pylint: disable=protected-access
    if device == "cuda" and pool_name == "page_unit_pool" and float(raf.build.with_cuda()) >= 11.3:
        pytest.skip(
            "Skip this because VM will use cudaAllocAsync to allocate memory. The "
            "underlying cuda memory pool is not compatible with raf page_unit_pool"
        )

    class Model(raf.Model):
        # pylint: disable=attribute-defined-outside-init,no-self-use
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            y = raf.conv2d(x, w, stride=1, padding=1, dilation=1, groups=1)
            y = raf.conv2d(y, w, stride=1, padding=1, dilation=1, groups=1)
            y = raf.conv2d(y, w, stride=1, padding=1, dilation=1, groups=1)
            return y

    InitPool(Device(device), pool_name)

    xshape = (32, 3, 224, 224)
    wshape = (3, 3, 3, 3)
    model = Model()
    model.infer_mode()
    m_x, _ = randn(xshape, device=device)
    m_w, _ = randn(wshape, device=device)

    mod = model._internal(m_x, m_w).mod
    # Enable memory planning with opt_level=3 to check if memory profiler reflects buffer sharing.
    with tvm.transform.PassContext(opt_level=3):
        raf.utils.memory_profiler.reset()
        raf.utils.memory_profiler.start()
        VMExecutor(mod, device).make_executor()(m_x, m_w)
        raf.utils.memory_profiler.stop()

    ret_map = raf.utils.memory_profiler.get_max_memory_info(raf.Device(device))
    peak_memory = ret_map["max_allocated"].value

    # The buffer size in MBs for an output tensor of the conv2d in the model.
    # Note that since it fits to the page size, we can allocate the exact size for it.
    buffer_size = (32 * 3 * 224 * 224) * 4 / 1048576

    if device == "cuda":
        if pool_name == "page_unit_pool" or float(raf.build.with_cuda()) >= 11.3:
            # Peak memory should have 2 tensors, but CuDNN Conv2D has workspace memory that
            # depends on the Conv2D algorithm selected by CuDNN.
            assert peak_memory >= 2 * buffer_size, "%.2f vs. %.2f" % (peak_memory, 2 * buffer_size)
        else:
            # CUDA 11.2- does not have CUDA memory pool, so the profiling results are always 0
            # with no_pool.
            assert peak_memory == 0
    else:
        if pool_name == "page_unit_pool":
            check(peak_memory, 2 * buffer_size, rtol=1e-1, atol=1e-1)
        else:
            assert peak_memory == 0


if __name__ == "__main__":
    pytest.main([__file__])
