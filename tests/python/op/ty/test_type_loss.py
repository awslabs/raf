# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
from raf._ffi.pass_ import AutoDiff, InferType
from raf.testing import check_type, run_infer_type, randn, one_hot_torch
from tvm.relay import TensorType, FuncType, TupleType


# pylint: disable=invalid-name, attribute-defined-outside-init, too-many-locals
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("shape", [[4, 4]])
@pytest.mark.parametrize("learning_rate", [0.01, 0.05])
@pytest.mark.parametrize("mu", [-1.24, 0.81])
def test_sgd(shape, dtype, learning_rate, mu):
    class Sgd(raf.Model):
        def build(self, learning_rate, mu):
            self._learning_rate = learning_rate
            self._mu = mu

        @raf.model.trace
        def forward(self, x, dx, v):
            return raf.sgd(x, dx, v, self._learning_rate, self._mu)

    model = Sgd(learning_rate, mu)
    # forward
    m_x, _ = randn(shape, dtype=dtype)
    m_dx, _ = randn(shape, dtype=dtype)
    m_v, _ = randn(shape, dtype=dtype)
    m_func = model._internal(m_x, m_dx, m_v).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty, x_ty, x_ty], TupleType([x_ty, x_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1],
        [3, 7],
    ],
)
def test_nll_loss(shape, dtype):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return raf.nll_loss(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    n, c = shape
    m_pred, _ = randn((n, c), dtype=dtype)
    m_true, _ = one_hot_torch(n, c)
    m_pred.requires_grad = True
    m_true.requires_grad = True
    ty_pred = TensorType((n, c), dtype=dtype)
    ty_true = TensorType((n,), dtype="int64")
    fwd_ty = TensorType((1,), dtype=dtype)
    # forward
    record = model._internal(m_true, m_pred)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([ty_true, ty_pred], fwd_ty)
    check_type(m_mod["main"], desired_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([ty_true, ty_pred]))
    desired_type = FuncType([ty_true, ty_pred], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [1, 3],
        [2, 3, 7],
    ],
)
@pytest.mark.parametrize("loss_type", ["cross_entropy", "smooth_l1_loss"])
def test_other_losses(loss_type, shape, dtype):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            if loss_type == "cross_entropy":
                loss = raf.cross_entropy(y_true=y_true, y_pred=y_pred)
            elif loss_type == "smooth_l1_loss":
                loss = raf.smooth_l1_loss(y_true=y_true, y_pred=y_pred)
            return loss

    model = TestModel()
    m_true, _ = randn(shape, dtype=dtype)
    m_pred, _ = randn(shape, dtype=dtype)
    m_true.requires_grad = True
    m_pred.requires_grad = True
    ty_pred = TensorType(shape, dtype=dtype)
    fwd_ty = TensorType((1,), dtype=dtype)
    # forward
    record = model._internal(m_pred, m_true)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([ty_pred, ty_pred], fwd_ty)
    check_type(m_mod["main"], desired_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([ty_pred, ty_pred]))
    desired_type = FuncType([ty_pred, ty_pred], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


if __name__ == "__main__":
    pytest.main([__file__])
