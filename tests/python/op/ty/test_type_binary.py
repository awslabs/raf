import pytest
import mnm
from mnm._op import sym
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType

@pytest.mark.parametrize("op", [
    sym.add,
    sym.subtract,
    sym.multiply,
    sym.divide,
    sym.mod,
])
@pytest.mark.parametrize("shape", [
    [(10, 4), (5, 10, 1), (5, 10, 4)],
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_binary(op, shape, dtype):
    # pylint: disable=too-many-locals
    class Binary(mnm.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    s_a, s_b, s_c = shape
    t_a = TensorType(s_a, dtype=dtype)
    t_b = TensorType(s_b, dtype=dtype)
    t_c = TensorType(s_c, dtype=dtype)
    # check forward
    m_a, _ = randn(s_a, dtype=dtype)
    m_b, _ = randn(s_b, dtype=dtype)
    m_func = model.get_relay_func(m_a, m_b)
    m_func = run_infer_type(m_func)
    desired_type = FuncType([t_a, t_b], t_c)
    check_type(m_func, desired_type)
    # TODO(@hzfan): check backward. needs to fold const for get_reduce_axis


@pytest.mark.parametrize("op", [
    sym.less,
    sym.greater,
    sym.less_equal,
    sym.greater_equal,
    sym.equal,
    sym.not_equal,
])
@pytest.mark.parametrize("shape", [
    [(10, 4), (5, 10, 1), (5, 10, 4)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logiacal(op, shape, dtype):
    # pylint: disable=too-many-locals
    class Binary(mnm.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    s_a, s_b, s_c = shape
    t_a = TensorType(s_a, dtype=dtype)
    t_b = TensorType(s_b, dtype=dtype)
    t_c = TensorType(s_c, dtype="bool")
    # check forward
    m_a, _ = randn(s_a, dtype=dtype)
    m_b, _ = randn(s_b, dtype=dtype)
    m_func = model.get_relay_func(m_a, m_b)
    m_func = run_infer_type(m_func)
    desired_type = FuncType([t_a, t_b], t_c)
    check_type(m_func, desired_type)
    # TODO(@hzfan): check backward. needs to fold const for get_reduce_axis


if __name__ == "__main__":
    pytest.main([__file__])
