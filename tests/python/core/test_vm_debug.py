import pytest
import mnm
from mnm.testing import check
from mnm._core.profiler_vm import VMProfilerExecutor
from mnm.testing import get_device_list, randn


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
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
    executor = VMProfilerExecutor(mod, device, cache_interm_tensors=True)
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


if __name__ == "__main__":
    pytest.main([__file__])
