import numpy as np
import pytest
import mnm
from mnm._lib import tvm, relay


def get_ctx_list():
    ret = ["cpu"]
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
def test_memory_alloc(ctx, shape):
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

    model_before = Model()
    model_before.infer_mode()
    m_x, _ = randn(shape, ctx=ctx)
    func = model_before.get_relay_func(m_x)
    mod = mnm._ffi.ir._make.Module({relay.GlobalVar("main"): func})
    mod = mnm._ffi.pass_.InferType(mod)
    target_name = ctx if ctx != 'cpu' else 'llvm'
    with tvm.target.Target(target_name):
        mod = mnm._ffi.pass_.ManifestAlloc(mod)
    text = mod['main'].astext()
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "invoke_op" in text

if __name__ == "__main__":
    pytest.main([__file__])
