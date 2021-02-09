# pylint: disable=no-self-use
import numpy as np
import pytest
import mnm
from mnm.testing import get_device_list, randn, check, run_vm_model


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_argsort(device, shape, axis, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.argsort(x, axis=axis, dtype=dtype)

    m_x, n_x = randn(shape, device=device)
    model = TestModel()
    m_out = model(m_x)
    v_out = run_vm_model(model, device, [m_x])
    np_out = np.argsort(n_x, axis).astype(dtype)
    check(m_out, np_out)
    check(v_out, np_out)


if __name__ == "__main__":
    pytest.main([__file__])
