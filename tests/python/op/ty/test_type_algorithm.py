import pytest
import mnm
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType


# pylint: disable=attribute-defined-outside-init
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_argsort(shape, axis, is_ascend, dtype):

    class Argsort(mnm.Model):
        def build(self, axis, is_ascend, dtype):
            self._axis = axis
            self._is_ascend = is_ascend
            self._dtype = dtype

        @mnm.model.trace
        def forward(self, data):
            return mnm.argsort(data, axis=self._axis,
                               is_ascend=self._is_ascend, dtype=self._dtype)

    model = Argsort(axis, is_ascend, dtype)
    # forward
    m_x, _ = randn(shape)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=m_x.dtype)
    y_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)

if __name__ == "__main__":
    pytest.main([__file__])
