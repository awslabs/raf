import pytest
import mnm
from mnm._lib import tvm
from mnm._core.module import IRModule
from mnm.testing import get_device_list, randn


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_memory_alloc(device, shape):
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
    m_x, _ = randn(shape, device=device)
    func = model_before._internal(m_x).mod['main']
    mod = IRModule.from_expr(func)
    mod = mnm._ffi.pass_.InferType(mod)
    target_name = device if device != 'cpu' else 'llvm'
    with tvm.target.Target(target_name):
        mod = mnm._ffi.pass_.ManifestAlloc(mod)
    mod = mnm._ffi.pass_.InferType(mod)
    text = mod['main'].astext()
    assert "alloc_storage" in text
    assert "alloc_tensor" in text
    assert "invoke_op" in text


if __name__ == "__main__":
    pytest.main([__file__])
