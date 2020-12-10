# pylint: disable=protected-access
import pytest
import numpy as np
import torch
import mnm
from mnm._ffi.pass_ import AutoDiff
from mnm.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType

def one_hot(batch_size, num_classes, ctx="cpu", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    t_x = torch.tensor(targets, requires_grad=False)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x


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
    m_func = model._internal(m_x, m_dx, m_v).func
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty, x_ty, x_ty], TupleType([x_ty, x_ty]))
    check_type(m_func, expected_type)

@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [
    [1, 1],
    [1, 3],
    [3, 7],
])
def test_nll_loss(shape, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return mnm.nll_loss(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    n, c = shape
    m_pred, _ = randn((n, c), dtype=dtype)
    m_true, _ = one_hot(n, c, dtype=dtype)
    ty_pred = TensorType((n, c), dtype=dtype)
    fwd_ty = TensorType((1,), dtype=dtype)
    # forward
    m_func = model._internal(m_pred, m_true).func
    m_func = run_infer_type(m_func)
    desired_type = FuncType([ty_pred, ty_pred], fwd_ty)
    check_type(m_func, desired_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    bwd_ty = FuncType([fwd_ty], TupleType([ty_pred, ty_pred]))
    desired_type = FuncType([ty_pred, ty_pred], TupleType([fwd_ty, bwd_ty]))
    check_type(m_func, desired_type)

@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [
    [3],
    [1, 3],
    [2, 3, 7],
])
@pytest.mark.parametrize("loss_type", ["cross_entropy", "smooth_l1_loss"])
def test_other_losses(loss_type, shape, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            if loss_type == "cross_entropy":
                loss = mnm.cross_entropy(y_true=y_true, y_pred=y_pred)
            elif loss_type == "smooth_l1_loss":
                loss = mnm.smooth_l1_loss(y_true=y_true, y_pred=y_pred)
            return loss

    model = TestModel()
    m_true, _ = randn(shape, dtype=dtype)
    m_pred, _ = randn(shape, dtype=dtype)
    ty_pred = TensorType(shape, dtype=dtype)
    fwd_ty = TensorType((1,), dtype=dtype)
    # forward
    m_func = model._internal(m_pred, m_true).func
    m_func = run_infer_type(m_func)
    desired_type = FuncType([ty_pred, ty_pred], fwd_ty)
    check_type(m_func, desired_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    bwd_ty = FuncType([fwd_ty], TupleType([ty_pred, ty_pred]))
    desired_type = FuncType([ty_pred, ty_pred], TupleType([fwd_ty, bwd_ty]))
    check_type(m_func, desired_type)

if __name__ == "__main__":
    pytest.main([__file__])
