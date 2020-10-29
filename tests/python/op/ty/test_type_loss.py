import pytest
import mnm
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=invalid-name, attribute-defined-outside-init
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [[4, 4]])
@pytest.mark.parametrize("learning_rate", [0.01, 0.05])
@pytest.mark.parametrize("mu", [-1.24, -0.47, 0.81])
def test_sgd(shape, dtype, learning_rate, mu):

    class Sgd(mnm.Model):
        def build(self, learning_rate, mu):
            self._learning_rate = learning_rate
            self._mu = mu

        @mnm.model.trace
        def forward(self, x, dx, v):
            return mnm.sgd(x, dx, v, self._learning_rate, self._mu)

    model = Sgd(learning_rate, mu)
    # forward
    m_x, _ = randn(shape, dtype=dtype)
    m_dx, _ = randn(shape, dtype=dtype)
    m_v, _ = randn(shape, dtype=dtype)
    m_func = model.get_relay_func(m_x, m_dx, m_v)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty, x_ty, x_ty], TupleType([x_ty, x_ty]))
    check_type(m_func, expected_type)

if __name__ == "__main__":
    pytest.main([__file__])
