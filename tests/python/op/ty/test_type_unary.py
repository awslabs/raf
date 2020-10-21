import pytest
import mnm
from mnm._ffi.pass_ import AutoDiff
from mnm._op import sym
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


@pytest.mark.parametrize("op", [
    (sym.relu, True),
    (sym.log, False),
    (sym.cos, False),
    (sym.tanh, True),
    (sym.sigmoid, True),
    (sym.copy, False),
    (sym.abs, False),
    (sym.ceil, False),
    (sym.floor, False),
    (sym.exp, False),
    (sym.erf, True),
    (sym.sqrt, True),
    (sym.atan, False),
    (sym.negative, False),
])
@pytest.mark.parametrize("shape", [
    (),
    (2,),
    (1, 3),
    (3, 7, 9),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unary(op, shape, dtype):
    op, backward = op

    class Unary(mnm.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, x):
            return op(x)

    model = Unary()
    fwd_ty = TensorType(shape, dtype=dtype)

    # forward
    m_x, _ = randn(shape, dtype=dtype)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)

    desired_type = FuncType([fwd_ty], fwd_ty)
    check_type(m_func, desired_type)

    # check backward
    if backward:
        m_func = AutoDiff(m_func)
        m_func = run_infer_type(m_func)
        bwd_ty = FuncType([fwd_ty], fwd_ty)
        desired_type = FuncType([fwd_ty], TupleType([fwd_ty, bwd_ty]))
        check_type(m_func, desired_type)


if __name__ == "__main__":
    pytest.main([__file__])
