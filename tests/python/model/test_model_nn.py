# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import raf
from raf.testing import check, randn_torch  # pylint: disable=E0401


# TODO(@were): allow affine=False
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("num_features", [1, 16])
@pytest.mark.parametrize("affine", [True])
@pytest.mark.parametrize("is_train", [False, True])
def test_model_batch_norm(num_features, affine, is_train):
    device = "cuda"
    m_x, t_x = randn_torch([8, num_features, 5, 6], device=device)
    m_m, _ = randn_torch([num_features], requires_grad=True, device=device)
    m_v, _ = randn_torch(
        [num_features], mean=1e-5, requires_grad=True, positive=True, device=device
    )
    if affine:
        m_w, _ = randn_torch([num_features], requires_grad=True, device=device)
        m_b, _ = randn_torch(
            [num_features], mean=1e-5, requires_grad=True, positive=True, device=device
        )
    model = raf.model.nn.BatchNorm(num_features=num_features, affine=affine)
    model.running_mean = m_m
    model.running_var = m_v
    if affine:
        model.w = m_w
        model.b = m_b
    # pylint: disable=no-member
    t_model = torch.nn.BatchNorm2d(num_features=num_features, affine=affine)
    t_model.running_mean[:] = torch.from_numpy(m_m.numpy())
    t_model.running_var[:] = torch.from_numpy(m_v.numpy())
    if affine:
        t_model.weight.data[:] = torch.from_numpy(m_w.numpy())
        t_model.bias.data[:] = torch.from_numpy(m_b.numpy())
    # pylint: enable=no-member
    if is_train:
        model.train_mode()
        t_model.train()
    else:
        model.infer_mode()
        t_model.eval()
    model.to(device=device)
    t_model.to(device=device)
    m_y = model(m_x)
    t_y = t_model(t_x)
    check(m_y, t_y)
    check(model.running_mean, t_model.running_mean)
    check(model.running_var, t_model.running_var)
    if affine:
        check(model.w, t_model.weight)
        check(model.b, t_model.bias)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("bias", [False, True])
def test_model_conv2d(stride, dilation, padding, bias):
    device = "cuda"
    m_x, t_x = randn_torch([8, 3, 32, 32], std=0.001, requires_grad=True, device=device)
    m_w, t_w = randn_torch([16, 3, 3, 3], std=0.01, requires_grad=True, device=device)
    if bias:
        m_b, t_b = randn_torch([16], std=0.001, requires_grad=True, device=device)
        t_b = t_b.unsqueeze(1).unsqueeze(2)
    model = raf.model.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
        bias=bias,
    )
    model.w = m_w
    if bias:
        model.b = m_b
    model.to(device=device)
    m_y = model(m_x)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    if bias:
        t_y = t_y + t_b
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("in_features", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("out_features", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("bias", [False, True])
def test_model_dense(batch_size, in_features, out_features, bias):
    device = "cuda"
    m_x, t_x = randn_torch([batch_size, in_features], requires_grad=True, device=device)
    m_w, _ = randn_torch([out_features, in_features], requires_grad=True, device=device)
    if bias:
        m_b, _ = randn_torch([out_features], requires_grad=True, device=device)

    # pylint: disable=no-member
    t_model = torch.nn.Linear(in_features, out_features, bias=bias)
    t_model.weight.data[:] = torch.from_numpy(m_w.numpy())
    if bias:
        t_model.bias.data[:] = torch.from_numpy(m_b.numpy())
    # pylint: enable=no-member
    model = raf.model.Linear(in_features=in_features, out_features=out_features, bias=bias)
    model.w = m_w
    if bias:
        model.b = m_b
    model.to(device=device)
    t_model.to(device=device)
    m_y = model(m_x)
    t_y = t_model(t_x)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


def _fake_test_relu():
    m_x, _ = randn_torch([1, 2, 3, 4], device="cpu", requires_grad=True)

    class ReLU(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return raf.relu(x)

    model = ReLU()
    model(m_x)


def _fake_test_conv2d():
    model = raf.model.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    )
    m_x, _ = randn_torch([8, 3, 32, 32], device="cpu", requires_grad=True)
    model(m_x)


def _fake_test_batch_norm():
    num_features = 128
    model = raf.model.BatchNorm(num_features=num_features, eps=1e-5, momentum=0.1, affine=True)
    m_x, _ = randn_torch([5, num_features, 3, 3], device="cpu", requires_grad=True)
    model(m_x)


def _fake_test_binary_add():
    m_x1, _ = randn_torch([1, 2, 3], device="cpu", requires_grad=True)
    m_x2, _ = randn_torch([2, 2, 3], device="cpu", requires_grad=True)

    class Add(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):  # pylint: disable=no-self-use
            return raf.add(x1, x2)

    model = Add()
    model(m_x1, m_x2)


if __name__ == "__main__":
    pytest.main([__file__])
