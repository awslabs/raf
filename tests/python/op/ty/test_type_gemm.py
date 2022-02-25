# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
from raf._ffi.pass_ import AutoDiff, InferType
from raf.testing import check_type, randn, randn_torch
from tvm.relay import TensorType, FuncType, TupleType


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 3),
        (3, 7, 9),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_dense(shape, dtype):
    # pylint: disable=no-member, too-many-locals
    class Dense(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            return raf.dense(m_a, m_b)

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
    check_type(m_mod["main"], desired_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 3),
        (3, 7, 9),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_matmul(shape, dtype, transpose_a, transpose_b):
    # pylint: disable=no-member, too-many-locals
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            raf_op = [[raf.matmul, raf.matmul_nt], [raf.matmul_tn, raf.matmul_tt]]
            raf_op = raf_op[transpose_a][transpose_b]
            return raf_op(m_a, m_b)

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
    check_type(m_mod["main"], desired_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2, 3),
        (5, 3, 7, 9),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_batch_matmul(shape, dtype, transpose_a, transpose_b):
    # pylint: disable=no-member, too-many-locals
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_a, m_b):  # pylint: disable=no-self-use
            raf_op = [
                [raf.batch_matmul, raf.batch_matmul_nt],
                [raf.batch_matmul_tn, raf.batch_matmul_tt],
            ]
            raf_op = raf_op[transpose_a][transpose_b]
            return raf_op(m_a, m_b)

    # initialize
    model = TestModel()
    b, n, m, k = shape
    m_a, _ = randn_torch(
        (b, n, k) if not transpose_a else (b, k, n), dtype=dtype, requires_grad=True
    )
    m_b, _ = randn_torch(
        (b, k, m) if not transpose_b else (b, m, k), dtype=dtype, requires_grad=True
    )
    m_a.requires_grad = True
    m_b.requires_grad = True

    fwd_ty = TensorType((b, n, m), dtype=dtype)
    a_ty = TensorType((b, n, k) if not transpose_a else (b, k, n), dtype=dtype)
    b_ty = TensorType((b, k, m) if not transpose_b else (b, m, k), dtype=dtype)
    # check forward
    record = model._internal(m_a, m_b)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([a_ty, b_ty], fwd_ty)
    check_type(m_mod["main"], desired_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([a_ty, b_ty]))
    desired_type = FuncType([a_ty, b_ty], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


if __name__ == "__main__":
    pytest.main([__file__])
