# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import raf
from raf.testing import get_testable_devices, randn_torch, check, run_vm_model, randint


def one_hot(batch_size, num_classes, device="cpu", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = raf.array(m_x, device=device)
    t_x = torch.tensor(targets, device=device, requires_grad=False)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize(
    "shape",
    [
        [3],
        [1, 2],
    ],
)
def test_smooth_l1_loss(device, shape):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return raf.smooth_l1_loss(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    m_pred, t_pred = randn_torch(shape, device=device, requires_grad=True)
    m_true, t_true = randn_torch(shape, device=device)
    # forward
    t_loss = F.smooth_l1_loss(t_pred, t_true)
    m_loss = model(y_true=m_pred, y_pred=m_true)
    v_loss = run_vm_model(model, device, [m_true, m_pred])
    check(m_loss, t_loss)
    check(v_loss, t_loss)
    # backward
    t_loss.backward()
    m_loss.backward()
    check(m_pred.grad, t_pred.grad)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("n", [3, 7])
@pytest.mark.parametrize("c", [2, 6])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("one_hot_label", [True, False])
def test_nll_loss(device, n, c, dtype, one_hot_label):
    if device == "cpu" and dtype == "float16":
        pytest.skip("PyTorch nll_loss does not support float16 when using CPU.")

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return raf.nll_loss(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    m_pred, t_pred = randn_torch((n, c), dtype=dtype, device=device, requires_grad=True)
    m_true, np_true = randint((n,), low=0, high=c, device=device, dtype="int64")
    if not one_hot_label:
        m_true = np.zeros((n, c), dtype=dtype)
        for i in range(n):
            m_true[i, np_true[i]] = 1
        m_true = raf.array(m_true, device=device)
    t_true = torch.tensor(np_true, device=device)
    # forward
    t_loss = F.nll_loss(t_pred, t_true)
    m_loss = model(y_true=m_true, y_pred=m_pred)
    v_loss = run_vm_model(model, device, [m_true, m_pred])
    check(m_loss, t_loss)
    check(v_loss, t_loss)
    # backward
    m_dy, t_dy = randn_torch((), device=device, dtype=dtype)
    t_loss.backward(t_dy)
    m_loss.backward(m_dy)
    rtol = 1e-5 if dtype == "float32" else 1e-3
    atol = 1e-5 if dtype == "float32" else 1e-3
    check(m_pred.grad, t_pred.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("c", [2, 4, 6])
def test_cross_entropy(device, n, c):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, y_true, y_pred):  # pylint: disable=no-self-use
            return raf.cross_entropy(y_true=y_true, y_pred=y_pred)

    model = TestModel()
    m_pred, t_pred = randn_torch((n, c), device=device, requires_grad=True)
    m_true, np_true = randint((n,), low=0, high=c, device=device, dtype="int64")
    t_true = torch.tensor(np_true, device=device)
    # forward
    t_loss = F.cross_entropy(t_pred, t_true)
    m_loss = model(y_true=m_true, y_pred=m_pred)
    v_loss = run_vm_model(model, device, [m_true, m_pred])
    check(m_loss, t_loss)
    check(v_loss, t_loss)
    # backward
    m_dy, t_dy = randn_torch((), device=device)
    t_loss.backward(t_dy)
    m_loss.backward(m_dy)
    check(m_pred.grad, t_pred.grad)


# TODO(@icemelon9): enable vm test in the future
@pytest.mark.parametrize(
    "shape",
    [
        (3, 16, 128, 128),
        (3, 16),
    ],
)
@pytest.mark.parametrize("device", get_testable_devices())
def test_broadcast_add(shape, device):
    m_a, t_a = randn_torch(shape, device=device, requires_grad=True)
    n = len(shape)
    m_b, t_b = randn_torch([shape[1]] + [1] * (n - 2), device=device, requires_grad=True)
    t_y = t_a + t_b
    m_y = raf.add(m_a, m_b)
    check(m_y, t_y)
    m_dy, t_dy = randn_torch(m_y.shape, device=device)
    t_y.backward(t_dy)

    def raf_add_dx(m_dy, m_x):
        axis = raf.get_reduce_axis(m_dy, m_x)
        keep = raf.get_kept_dims(m_dy, m_x)
        return raf.sum(m_dy, axis=axis, keepdims=keep)

    m_da = raf_add_dx(m_dy, m_a)
    check(m_da, t_a.grad, atol=0.01 if n == 4 else 1e-4)
    m_db = raf_add_dx(m_dy, m_b)
    check(m_db, t_b.grad, atol=0.01 if n == 4 else 1e-4)


# TODO(@icemelon9): enable vm test in the future
@pytest.mark.parametrize("shape", [[4, 4]])
def test_sgd(shape):
    x0 = np.random.randn(*shape).astype("float32")
    dx = np.random.randn(*shape).astype("float32")
    v0 = np.random.randn(*shape).astype("float32")
    mu = np.random.randn(1)[0]  # pylint: disable=invalid-name
    learning_rate = 0.01

    n_v1 = mu * v0 + dx
    n_x1 = x0 - learning_rate * n_v1

    x0 = raf.array(x0)
    dx = raf.array(dx)
    v0 = raf.array(v0)
    m_v1, m_x1 = raf.sgd(x0, dx, v0, learning_rate, mu)

    np.testing.assert_allclose(m_v1.numpy(), n_v1, 1e-4, 1e-4)
    np.testing.assert_allclose(m_x1.numpy(), n_x1, 1e-4, 1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
