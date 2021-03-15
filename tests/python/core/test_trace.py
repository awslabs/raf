# pylint: disable=attribute-defined-outside-init, no-self-use
import pytest

import mnm
from mnm.testing import randn, get_device_list, check


@pytest.mark.parametrize("device", get_device_list())
def test_tup_inputs(device):
    class MNMTupleTest(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, tup):
            x = mnm.add(tup[0], tup[1])
            return x

    shape = (2, 2)
    m_model = MNMTupleTest()
    m_x, n_x = randn(shape, device=device)
    m_y, n_y = randn(shape, device=device)
    m_z = m_model((m_x, m_y))
    n_z = n_x + n_y
    check(m_z, n_z)


if __name__ == "__main__":
    pytest.main([__file__])
