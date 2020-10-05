import pytest
import numpy as np
import mnm
from mnm._core.executor import VMExecutor
from mnm._core.module import Module
import tvm

def get_ctx_list():
    ret = ["llvm"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    m_x.requires_grad = True
    return m_x, n_x

@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_vm(ctx, shape):
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
    m_x, _ = randn(shape, ctx=ctx)
    mod = Module()
    func = model.get_relay_func(m_x)
    mod[tvm.ir.GlobalVar('main')] = func
    executor = VMExecutor(mod, ctx)
    m_z = executor.make_executor()(m_x).asnumpy()
    ref_z = model(m_x).asnumpy()
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_tuple(ctx, shape):
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
    m_x, _ = randn(shape, ctx=ctx)
    mod = Module()
    func = model.get_relay_func(m_x)
    mod[tvm.ir.GlobalVar('main')] = func
    executor = VMExecutor(mod, ctx)
    m_y, m_z = executor.make_executor()(m_x)
    m_y, m_z = m_y.asnumpy(), m_z.asnumpy()
    ref_y, ref_z = model(m_x)
    ref_y, ref_z = ref_y.asnumpy(), ref_z.asnumpy()
    np.testing.assert_allclose(m_y, ref_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(m_z, ref_z, rtol=1e-5, atol=1e-5)

    executable = executor.executable
    assert len(executable.globals) == 1
    assert executable.globals[0] == 'main'


if __name__ == "__main__":
    pytest.main([__file__])
