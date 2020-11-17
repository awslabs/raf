import pytest
import mnm
from mnm._op import sym
from mnm._ffi.pass_ import AutoDiff
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


@pytest.mark.parametrize("op", [
    sym.add,
    sym.subtract,
    sym.multiply,
    sym.divide,
    sym.mod,
    sym.maximum,
    sym.minimum,
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
    # check backward
    # TODO(yzhliu): some operators are missing gradient registries.
    if op not in (sym.mod, sym.maximum, sym.minimum, sym.subtract):
        bwd_func = AutoDiff(m_func)
        bwd_func = run_infer_type(bwd_func)
        bwd_ty = FuncType([t_c], TupleType([t_a, t_b]))
        desired_type = FuncType([t_a, t_b], TupleType([t_c, bwd_ty]))
        check_type(bwd_func, desired_type)


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
    # TODO(@hzfan): check backward. missing gradient registries.


if __name__ == "__main__":
    pytest.main([__file__])
