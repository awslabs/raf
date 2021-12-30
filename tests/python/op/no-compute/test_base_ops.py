# pylint: disable=attribute-defined-outside-init,no-member,protected-access
import pytest
import numpy as np

import mnm
from mnm.testing import run_vm_model, randn, get_testable_devices


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("device", get_testable_devices())
def test_size(shape, axis, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self, axis):
            self.axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.size(x, self.axis)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.size(m_x, axis)
    n_y = np.array(n_x.shape[axis], dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model(axis=axis)
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_numel(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.numel(x)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.numel(m_x)
    n_y = np.array(n_x.size, dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model()
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_shape_as_tensor(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.shape_as_tensor(x)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.shape_as_tensor(m_x)
    n_y = np.array(n_x.shape, dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model()
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


if __name__ == "__main__":
    pytest.main([__file__])
