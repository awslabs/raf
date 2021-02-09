import numpy as np
import pytest
import mnm
from mnm.testing import get_device_list, randn, check


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_add_to(shape, device):
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm.add(x, x)
    model = Add()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, device=device)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 2 * n_dy
    check(m_dx, n_dx)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3,],
    [4,]
])
def test_no_grad(shape, device):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y, z):  # pylint: disable=no-self-use
            indices = mnm.add(y, z)
            indices = mnm.subtract(indices, z)
            indices = mnm.add(indices, z)
            return mnm.take(x, indices, axis=0)

    model = Model()
    m_x, n_x = randn(shape, device=device, requires_grad=True)
    m_y = mnm.array([1,], dtype="int64", device=device)
    m_z = mnm.array([1,], dtype="int64", device=device)
    m_out = model(m_x, m_y, m_z)  # m_out = m_x[2]
    m_dout, n_dout = randn([1,], device=device)
    m_out.backward(m_dout)
    m_dx = m_x.grad
    n_dx = np.zeros_like(n_x)
    n_dx[2] = n_dout[0]
    check(m_dx, n_dx)


if __name__ == "__main__":
    pytest.main([__file__])
