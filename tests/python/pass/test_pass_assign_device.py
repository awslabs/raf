
# pylint: disable=unused-import, attribute-defined-outside-init, protected-access
# pylint: disable=missing-module-docstring, missing-function-docstring, no-self-use
import pytest
import mnm
from mnm.frontend import FrameworkModel

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_init():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self):
            return mnm.ones(shape=(3, 2), dtype="float32", device="cpu")

    m_model = Model()
    mod = m_model._internal().mod

    # Assign device now only applies to FrameworkModel.
    m_model = FrameworkModel(mod, mod, {}, {})
    m_model.to(device="cuda")
    out = m_model()
    assert out.device.startswith("cuda")

if __name__ == "__main__":
    pytest.main([__file__])
