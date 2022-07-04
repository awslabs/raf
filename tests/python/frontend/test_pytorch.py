# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint:disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# pylint:disable=not-callable,abstract-method,too-many-locals,invalid-name,protected-access
# pylint: disable=too-many-statements
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf._op import sym
from raf.frontend import from_pytorch
from raf.testing import randn_torch, check, one_hot_torch, run_vm_model, with_seed


class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "shape_dict",
    [{"input0": ((32, 3, 28, 28), "float32")}, {"input0": ((32, 3, 28, 28), "float16")}],
)
@pytest.mark.parametrize("mode", ["forward", "backward", "sgd"])
@with_seed(0)
def test_lenet(shape_dict, mode):
    device = "cuda"
    input_shape, dtype = list(shape_dict.values())[0]
    batch_size = input_shape[0]

    # Prepare two models.
    t_model = TorchLeNet(input_shape[2])
    if dtype == "float16":
        t_model = t_model.half()
    m_model = from_pytorch(t_model, shape_dict)

    # Set the target device.
    t_model.to(device=device)
    m_model.to(device=device)

    # Prepare data.
    m_x, t_x = randn_torch(input_shape, device=device, dtype=dtype)

    if mode == "forward":
        m_model.infer_mode()
        t_model.eval()
        m_y = m_model(m_x)
        t_y = t_model(t_x)
        tol = 1e-4 if dtype == "float32" else 1e-3
        check(m_y, t_y, rtol=tol, atol=tol)
        return

    m_ytrue, t_ytrue = one_hot_torch(size=batch_size, num_classes=10, device=device)
    m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False, dtype=dtype)

    # append loss function
    out = m_model.record(m_x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    m_model = m_model + loss

    if mode == "backward":
        m_x.requires_grad = True

        m_model.train_mode()
        t_model.train()

        m_loss = m_model(m_x, m_ytrue)

        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)

        tol = 1e-5 if dtype == "float32" else 1e-2
        check(m_loss, t_loss, rtol=tol, atol=tol)

        m_loss.backward()
        t_loss.backward()
        tol = 1e-4 if dtype == "float32" else 1e-2
        check(m_loss, t_loss, rtol=tol, atol=tol)
    else:
        assert mode == "sgd"

        m_model.train_mode()
        m_model.to(device=device)

        m_trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
        m_loss = run_vm_model(m_trainer, device, [m_dy, m_x, m_ytrue])[0]

        t_trainer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
        t_model.train()

        t_trainer.zero_grad()
        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)
        t_loss.backward(t_dy)
        t_trainer.step()
        tol = 1e-5 if dtype == "float32" else 1e-2
        check(m_loss, t_loss, rtol=tol, atol=tol)


class TorchConvBn(nn.Module):
    def __init__(self):
        super(TorchConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.bn = torch.nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(x.shape[0], -1)
        return x


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape_dict", [{"input0": ((32, 3, 28, 28), "float32")}])
@pytest.mark.parametrize("mode", ["backward", "sgd"])
@pytest.mark.parametrize("fuse", [False, True])
@with_seed(0)
def test_conv_bn(shape_dict, mode, fuse):
    device = "cuda"
    input_shape = list(shape_dict.values())[0][0]
    batch_size = input_shape[0]

    # Prepare two models.
    t_model = TorchConvBn()
    m_model = from_pytorch(t_model, shape_dict)

    # Set the target device.
    t_model.to(device=device)
    m_model.to(device=device)

    # Prepare data.
    m_x, t_x = randn_torch(input_shape, device=device)

    if mode == "forward":
        m_model.infer_mode()
        t_model.eval()
        m_y = m_model(m_x)
        t_y = t_model(t_x)
        check(m_y, t_y, rtol=1e-4, atol=1e-4)
        return

    m_ytrue, t_ytrue = one_hot_torch(size=batch_size, num_classes=6 * 24 * 24, device=device)
    m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False)

    # append loss function
    out = m_model.record(m_x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    m_model = m_model + loss

    if mode == "backward":
        m_x.requires_grad = True

        m_model.train_mode()
        t_model.train()

        m_loss = m_model(m_x, m_ytrue)[0]

        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)

        check(m_loss, t_loss, rtol=1e-4, atol=1e-4)

        m_loss.backward()
        t_loss.backward()
        check(m_loss, t_loss, rtol=1e-4, atol=1e-4)
    else:
        assert mode == "sgd"

        m_model.train_mode()
        m_model.to(device=device)

        m_trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
        m_loss = run_vm_model(m_trainer, device, [m_dy, m_x, m_ytrue], disable_fusion=not fuse)[0][
            0
        ]

        t_trainer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
        t_model.train()

        t_trainer.zero_grad()
        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)
        t_loss.backward(t_dy)
        t_trainer.step()
        check(m_loss, t_loss, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape_dict", [{"input0": ((32, 3, 28, 28), "float32")}])
@with_seed(0)
def test_batch_norm_train(shape_dict):
    device = "cuda"
    input_shape = list(shape_dict.values())[0][0]

    # Prepare two models.
    t_model = nn.BatchNorm2d(input_shape[1])
    m_model = from_pytorch(t_model, shape_dict)

    # Set the target device.
    t_model.to(device=device)
    m_model.to(device=device)

    # Prepare data.
    m_x, t_x = randn_torch(input_shape, device=device)

    m_model.train_mode()
    t_model.train()
    m_y = run_vm_model(m_model, device, [m_x])[0]
    t_y = t_model(t_x)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)
    check(m_model.state()["model_running_mean"], t_model.running_mean)
    check(m_model.state()["model_running_var"], t_model.running_var)


class TorchMatmulDropout(nn.Module):
    def __init__(self, p):
        super(TorchMatmulDropout, self).__init__()
        self.linear = nn.Linear(20, 30)
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        return x


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape_dict", [{"input0": ((128, 20), "float32")}])
@pytest.mark.parametrize("p", [0.6, 0.0])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("mode", ["forward", "backward", "sgd"])
@with_seed(0)
def test_mm_dropout(shape_dict, p, device, mode):
    input_shape, dtype = list(shape_dict.values())[0]
    batch_size = input_shape[0]

    # Prepare two models.
    t_model = TorchMatmulDropout(p)
    m_model = from_pytorch(t_model, shape_dict)

    # Set the target device.
    t_model.to(device=device)
    m_model.to(device=device)

    # Prepare data.
    m_x, t_x = randn_torch(input_shape, device=device)
    m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False, dtype=dtype)

    if mode == "forward":
        m_model.infer_mode()
        t_model.eval()
        m_y = m_model(m_x)
        t_y = t_model(t_x)
        if p == 0.0:
            check(m_y, t_y, rtol=1e-4, atol=1e-4)
        return

    m_ytrue, t_ytrue = one_hot_torch(size=batch_size, num_classes=30, device=device)

    # append loss function
    out = m_model.record(m_x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    m_model = m_model + loss

    if mode == "backward":
        m_x.requires_grad = True

        m_model.train_mode()
        t_model.train()

        m_loss = m_model(m_x, m_ytrue)

        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)

        if p == 0.0:
            check(m_loss, t_loss, rtol=1e-4, atol=1e-4)

        m_loss.backward()
        t_loss.backward()
        if p == 0.0:
            check(m_loss, t_loss, rtol=1e-4, atol=1e-4)
    else:
        assert mode == "sgd"

        m_model.train_mode()
        m_model.to(device=device)

        m_trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
        m_loss = run_vm_model(m_trainer, device, [m_dy, m_x, m_ytrue], disable_fusion=True)[0]

        t_trainer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
        t_model.train()

        t_trainer.zero_grad()
        t_y = t_model(t_x)
        t_ypred = torch.log_softmax(t_y, dim=-1)
        t_loss = F.nll_loss(t_ypred, t_ytrue)
        t_loss.backward(t_dy)
        t_trainer.step()
        if p == 0.0:
            check(m_loss, t_loss, rtol=1e-4, atol=1e-4)


def test_params_order():
    class TorchConv(nn.Module):
        def __init__(self, p, shape):
            super(TorchConv, self).__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
            self.p = p
            A = torch.randn(shape, requires_grad=True)
            self.A = torch.nn.Parameter(A)

        def forward(self, x):
            b = self.p + self.A
            x = x + b
            x = self.conv(x)
            return x

    device = "cpu"
    shape_dict = {"input0": ((32, 3, 28, 28), "float32")}
    input_shape = list(shape_dict.values())[0][0]

    t_model = TorchConv(1.0, input_shape)
    m_model = from_pytorch(t_model, shape_dict)

    t_model.to(device=device)
    m_model.to(device=device)

    m_x, _ = randn_torch(input_shape, device=device)
    out = m_model.record(m_x)
    re = sym.relu(out)
    m_model = m_model + re


@pytest.mark.parametrize("shape_dict", [{"input0": ((32, 3, 28, 28), "float32")}])
def test_save_and_load_model(shape_dict):
    input_shape = list(shape_dict.values())[0][0]

    t_model = TorchLeNet(input_shape[2])
    with tempfile.TemporaryDirectory(prefix="raf_test_") as temp_dir:
        model_path = temp_dir + "/test.pt"
        hash_path = temp_dir + "/test.hash"

        # Test save model
        from_pytorch(t_model, shape_dict, model_path, hash_path)
        # Test load model
        from_pytorch(t_model, shape_dict, model_path, hash_path)


def test_learnable_params():
    class TorchModel(nn.Module):
        def __init__(self, shape):
            super(TorchModel, self).__init__()
            A = torch.randn(shape, requires_grad=True)
            B = torch.randn(shape, requires_grad=True)
            self.A = torch.nn.Parameter(A)
            self.B = torch.nn.Parameter(B, requires_grad=False)
            self.register_buffer("buffer", torch.zeros(shape))

        def forward(self, x):
            b = x + self.A
            y = b + self.B
            return y + self.buffer

    device = "cpu"
    shape_dict = {"input0": ((32, 3, 28, 28), "float32")}
    input_shape = list(shape_dict.values())[0][0]

    t_model = TorchModel(input_shape)
    m_model = from_pytorch(t_model, shape_dict)

    t_model.to(device=device)
    m_model.to(device=device)
    m_model.train_mode()
    m_params = m_model.state()
    for n, param in t_model.named_parameters():
        assert m_params["model_" + n].requires_grad == param.requires_grad
    assert not m_params["model_buffer"].requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
