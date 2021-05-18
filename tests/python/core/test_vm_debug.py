import pytest
import mnm
from mnm.testing import check
from mnm._core.profiler_vm import VMProfilerExecutor
from mnm.testing import get_device_list, randn, with_seed


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
@with_seed(0)
def test_vm_debug(device, shape):
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(x, x)
            z = mnm.add(x, y)
            return z

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    mod = model._internal(m_x).mod
    # disable fusion
    with mnm.ir.PassContext(opt_level=1):
        executor = VMProfilerExecutor(mod, device, cache_interm_tensors=True)

    # Testing whether we can get the correct intermediate tensor
    m_z = executor.make_executor()(m_x).asnumpy()
    ref_x = m_x.asnumpy()
    ref_y = ref_x + ref_x
    ref_z = model(m_x).asnumpy()
    check(m_z, ref_z, rtol=1e-5, atol=1e-5)
    _, ins, outs = executor.get_interm_tensors()
    check(ins[0][0], ref_x)
    check(ins[0][1], ref_x)
    check(ins[1][0], ref_x)
    check(ins[1][1], ref_y)
    check(outs[0], ref_y)
    check(outs[1], ref_z)

@pytest.mark.parametrize("device", get_device_list())
def test_vm_memory_profile(device):
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init,no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w):
            return mnm.conv2d(x, w, stride=1, padding=0, dilation=1, groups=1)

    xshape = (32, 3, 224, 224)
    wshape = (32, 3, 3, 3)
    model = Model()
    model.infer_mode()
    m_x, _ = randn(xshape, device=device)
    m_w, _ = randn(wshape, device=device)

    mod = model._internal(m_x, m_w).mod
    executor = VMProfilerExecutor(mod, device)
    ret_map = executor.make_executor()(m_x, m_w, profile_memory=True)
    ret = sum([v.value for _, v in ret_map.items()])

    # Output tensor size in MBs. Note that since it fits to the page size,
    # we can allocate the exact size for it.
    expected = (32 * 32 * 222 * 222) * 4 / 1048576.0
    if device == "cuda":
        # Conv2D CuDNN requests an additional workspace memory but its size depends on the
        # CuDNN implementation. To avoid flaky test we simply check whether memory profiler
        # catches the workspace request.
        assert ret > expected
    else:
        check(ret, expected)


if __name__ == "__main__":
    pytest.main([__file__])
