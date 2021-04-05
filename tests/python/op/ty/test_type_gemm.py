# pylint: disable=protected-access
import pytest
import mnm
from mnm._ffi.pass_ import AutoDiff, InferType
from mnm.testing import check_type, randn, randn_torch
from tvm.relay import TensorType, FuncType, TupleType


@pytest.mark.parametrize("shape", [
    (1, 2, 3),
    (1, 5, 7),
    (3, 7, 9),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dense(shape, dtype):
    # pylint: disable=no-member, too-many-locals
    class Dense(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            return mnm.dense(m_a, m_b)
    # initialize
    model = Dense()
    n, m, k = shape
    fwd_ty = TensorType((m, n), dtype=dtype)
    a_ty = TensorType((m, k), dtype=dtype)
    b_ty = TensorType((n, k), dtype=dtype)
    m_a, _ = randn((m, k), dtype=dtype)
    m_b, _ = randn((n, k), dtype=dtype)
    m_a.requires_grad = True
    m_b.requires_grad = True
    # check forward
    record = model._internal(m_a, m_b)
    m_mod = record.mod
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([a_ty, b_ty], fwd_ty)
    check_type(m_mod['main'], desired_type)
    # check backward
    m_mod = AutoDiff(m_mod, record.requires_grads)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod['main'], desired_type)

@pytest.mark.parametrize("shape", [
    (1, 2, 3),
    (1, 5, 7),
    (3, 7, 9),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_matmul(shape, dtype, transpose_a, transpose_b):
    # pylint: disable=no-member, too-many-locals
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            mnm_op = [[mnm.matmul, mnm.matmul_nt],
                      [mnm.matmul_tn, mnm.matmul_tt]]
            mnm_op = mnm_op[transpose_a][transpose_b]
            return mnm_op(m_a, m_b)
    # initialize
    model = TestModel()
    n, m, k = shape
    m_a, _ = randn_torch((n, k) if not transpose_a else (k, n), dtype=dtype, requires_grad=True)
    m_b, _ = randn_torch((k, m) if not transpose_b else (m, k), dtype=dtype, requires_grad=True)
    m_a.requires_grad = True
    m_b.requires_grad = True
    fwd_ty = TensorType((n, m), dtype=dtype)
    a_ty = TensorType((n, k) if not transpose_a else (k, n), dtype=dtype)
    b_ty = TensorType((k, m) if not transpose_b else (m, k), dtype=dtype)
    # check forward
    record = model._internal(m_a, m_b)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([a_ty, b_ty], fwd_ty)
    check_type(m_mod['main'], desired_type)
    # check backward
    m_mod = AutoDiff(m_mod, record.requires_grads)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod['main'], desired_type)

@pytest.mark.parametrize("shape", [
    (1, 1, 2, 3),
    (1, 3, 7, 9),
    (3, 1, 2, 3),
    (5, 3, 7, 9),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batch_matmul(shape, dtype):
    # pylint: disable=no-member, too-many-locals
    class BatchMatmul(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            return mnm.batch_matmul(m_a, m_b)
    # initialize
    model = BatchMatmul()
    b, n, m, k = shape
    m_a, _ = randn((b, m, k), dtype=dtype)
    m_b, _ = randn((b, n, k), dtype=dtype)
    m_a.requires_grad = True
    m_b.requires_grad = True
    fwd_ty = TensorType((b, m, n), dtype=dtype)
    a_ty = TensorType((b, m, k), dtype=dtype)
    b_ty = TensorType((b, n, k), dtype=dtype)
    # check forward
    record = model._internal(m_a, m_b)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([a_ty, b_ty], fwd_ty)
    check_type(m_mod['main'], desired_type)
    # check backward
    m_mod = AutoDiff(m_mod, record.requires_grads)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod['main'], desired_type)


if __name__ == "__main__":
    pytest.main([__file__])
