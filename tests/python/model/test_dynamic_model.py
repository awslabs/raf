import numpy as np
import pytest
import mnm
from mnm.testing import get_testable_devices, check, run_vm_model


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("fuse", [True, False])
def test_dynamic_model(device, fuse):
    # pylint: disable=no-self-use
    class Model(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            y = mnm.argwhere(x)
            y = mnm.split(y, 2)
            y = mnm.add(y[0], y[1])
            y = mnm.abs(y)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = mnm.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x], disable_fusion=not fuse)
    expected = mnm.array([[1, 0], [1, 2]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", get_testable_devices())
def test_dynamic_reshape(device):
    # pylint: disable=no-self-use
    class Model(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):
            y = mnm.argwhere(x)
            y = mnm.split(y, 2)
            y = mnm.add(y[0], y[1])
            y = mnm.abs(y)
            y = mnm.expand_dims(y, 0)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = mnm.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x])
    expected = mnm.array([[[1, 0], [1, 2]]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


if __name__ == "__main__":
    pytest.main([__file__])
