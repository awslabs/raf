import pytest
import numpy as np
import mnm
from mnm.testing import run_infer_type, run_vm_model, check
from mnm._core.executor import VMExecutor
from mnm.testing import get_arr_addr, get_device_list, randn


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_vm(device, shape):
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
    executor = VMExecutor(mod, device)
    m_z = executor.make_executor()(m_x).asnumpy()
    ref_z = model(m_x).asnumpy()
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'

    # execute 2nd time to reuse the op env
    m_z = executor.vm.run(m_x).asnumpy()
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_cuda_graph(shape):
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

    dev = "cuda"
    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=dev)
    mod = model._internal(m_x).mod
    executor = VMExecutor(mod, dev, enable_cuda_graph=True)
    m_z = executor.make_executor()(m_x)
    ref_z = model(m_x).asnumpy()
    np.testing.assert_allclose(m_z.asnumpy(), ref_z, rtol=1e-5, atol=1e-5)

    m_x2, _ = randn(shape, device=dev)
    m_z2 = executor.vm.run(m_x2)
    ref_z2 = model(m_x2).asnumpy()
    np.testing.assert_allclose(m_z2.asnumpy(), ref_z2, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_tuple(device, shape):
    # pylint: disable=protected-access
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(x, x)
            z = mnm.add(x, y)
            return y, z

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    mod = model._internal(m_x).mod
    executor = VMExecutor(mod, device)
    m_y, m_z = executor.make_executor()(m_x)
    m_y, m_z = m_y.asnumpy(), m_z.asnumpy()
    ref_y, ref_z = model(m_x)
    ref_y, ref_z = ref_y.asnumpy(), ref_z.asnumpy()
    np.testing.assert_allclose(m_y, ref_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_memory(device, shape):
    # pylint: disable=protected-access
    dtype = 'float32'
    x = mnm.array(np.random.randn(*shape).astype(dtype), device=device)
    t_1 = mnm.array(np.ones(shape, dtype=dtype) * 3)
    t_2 = mnm.array(np.ones(shape, dtype=dtype) * 4)
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.relu(x)
            return y

    model = Model()
    args = [x]
    mod = model._internal(*args).mod
    executor = VMExecutor(mod, device)
    y = executor.make_executor()(*args)
    out = mnm.add(t_1, t_2)
    assert get_arr_addr(out) != get_arr_addr(y)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_simple_fusion(device, shape):
    # pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
    def ir_fusion(mod):
        mod = run_infer_type(mod)
        mod = mnm._ffi.pass_.FuseOps(mod, 3)
        mod = run_infer_type(mod)
        return mod

    def check_e2e(model, device, args):
        out_before = run_vm_model(model, device, args)
        out_after = run_vm_model(model, device, args, ir_fusion)
        check(out_before, out_after)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, x)
            z = mnm.relu(y)
            return z

    model = Model()
    model.infer_mode()
    m_x, _ = randn(shape, device=device)
    check_e2e(model, device, [m_x])


@pytest.mark.parametrize("device", get_device_list())
def test_split_fusion(device):
    # pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
    shape = [3, 3]
    def ir_fusion(mod):
        mod = run_infer_type(mod)
        mod = mnm._ffi.pass_.FuseOps(mod, 3)
        mod = run_infer_type(mod)
        return mod

    def check_e2e(model, device, args):
        out_before = run_vm_model(model, device, args)
        out_after = run_vm_model(model, device, args, ir_fusion)
        check(out_before, out_after)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.split(x, indices_or_sections=3, axis=0)
            y = y[0]
            z = mnm.relu(y)
            return z

    model = Model()
    m_x, _ = randn(shape, device=device)
    check_e2e(model, device, [m_x])


if __name__ == "__main__":
    pytest.main([__file__])
