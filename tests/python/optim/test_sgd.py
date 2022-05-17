# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import re

from unittest.mock import patch
import pytest
import numpy as np

import mxnet as mx
from mxnet import gluon
import torch
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf.model import Conv2d, Linear, BatchNorm
from raf.testing import (
    with_seed,
    get_testable_devices,
    check,
    run_vm_model,
    one_hot_torch,
    randn_torch,
    t2m_param,
    randn_mxnet,
)


class TorchTest(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchTest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(6)
        self.linear1 = nn.Linear((input_shape // 2) ** 2 * 6, num_classes)

    def forward(self, x, y_true):  # pylint: disable=arguments-differ
        y_pred = self.forward_infer(x)
        y_pred = F.log_softmax(y_pred, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss

    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = torch.sigmoid(out)  # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)  # pylint: disable=no-member
        out = self.linear1(out)
        return out


class RAFTest(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=True)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

    # pylint: enable=attribute-defined-outside-init

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = raf.sigmoid(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = raf.batch_flatten(out)
        out = self.linear1(out)
        return out


# pylint: disable=unused-variable
@with_seed(0)
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("config", [(10, 32, 10)])
def test_sgd(config):
    t_model = TorchTest(config[1], config[2])
    t_model.to(device="cuda")
    m_model = RAFTest(config[1], config[2])
    m_model.to(device="cuda")
    m_model.conv1.w = t2m_param(t_model.conv1.weight)
    m_model.conv1.b = t2m_param(t_model.conv1.bias)
    m_model.linear1.w = t2m_param(t_model.linear1.weight)
    m_model.linear1.b = t2m_param(t_model.linear1.bias)
    m_model.bn1.w = t2m_param(t_model.bn1.weight)
    m_model.bn1.b = t2m_param(t_model.bn1.bias)
    m_model.bn1.running_mean = t2m_param(t_model.bn1.running_mean)
    m_model.bn1.running_var = t2m_param(t_model.bn1.running_var)

    m_param_dict = m_model.state()
    m_optimizer = raf.optim.SGD(m_param_dict.values(), 0.1, 0.01)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    batch_size = config[0]
    m_model.train_mode()
    t_model.train()

    for i in range(batch_size):
        t_optimizer.zero_grad()
        m_x, t_x = randn_torch([1, 3, config[1], config[1]], requires_grad=True, device="cuda")
        m_y, t_y = one_hot_torch(size=1, num_classes=config[2], device="cuda")
        m_loss = m_model(m_x, m_y)
        t_loss = t_model(t_x, t_y)
        m_loss.backward()
        t_loss.backward()
        check(m_model.conv1.w.grad, t_model.conv1.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.conv1.b.grad, t_model.conv1.bias.grad, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.w.grad, t_model.linear1.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.b.grad, t_model.linear1.bias.grad, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.w.grad, t_model.bn1.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.b.grad, t_model.bn1.bias.grad, rtol=1e-4, atol=1e-4)
        check(m_loss, t_loss)
        m_optimizer.step()
        t_optimizer.step()
        check(m_model.conv1.w, t_model.conv1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.w, t_model.linear1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.b, t_model.linear1.bias, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.w, t_model.bn1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.b, t_model.bn1.bias, rtol=1e-4, atol=1e-4)


class TorchSimpleTest(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, shape):
        super(TorchSimpleTest, self).__init__()
        self.x = torch.nn.Parameter(torch.randn(*shape))
        self.x.requires_grad = True

    def forward(self):  # pylint: disable=arguments-differ
        y = F.relu(self.x)
        return y


class RAFSimpleTest(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, shape):
        self.x = raf.array(np.random.randn(*shape))

    @raf.model.trace
    def forward(self):
        y = raf.relu(self.x)
        return y


@pytest.mark.parametrize("device", get_testable_devices())
def test_traced_sgd_simple(device):
    # pylint: disable=attribute-defined-outside-init
    shape = (2, 2)
    batch_size = 32
    dtype = "float32"
    t_model = TorchSimpleTest(shape)
    t_model.to(device)
    m_model = RAFSimpleTest(shape)
    m_model.x = t2m_param(t_model.x, device=device)
    m_model.train_mode()
    t_model.train()
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    for i in range(batch_size):
        m_dy, t_dy = randn_torch(shape, device=device, requires_grad=False)
        m_loss = run_vm_model(m_optimizer, device, [m_dy])
        t_optimizer.zero_grad()
        t_loss = t_model()
        t_loss.backward(t_dy)
        t_optimizer.step()
        check(m_model.x, t_model.x, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("config", [(10, 32, 10)])
def test_traced_sgd(config):
    # pylint: disable=too-many-locals
    device = "cuda"
    t_model = TorchTest(config[1], config[2])
    t_model.to(device=device)
    m_model = RAFTest(config[1], config[2])
    m_model.to(device=device)
    m_model.conv1.w = t2m_param(t_model.conv1.weight, device=device)
    m_model.linear1.w = t2m_param(t_model.linear1.weight, device=device)
    m_model.linear1.b = t2m_param(t_model.linear1.bias, device=device)
    m_model.bn1.w = t2m_param(t_model.bn1.weight, device=device)
    m_model.bn1.b = t2m_param(t_model.bn1.bias, device=device)
    m_model.bn1.running_mean = t2m_param(t_model.bn1.running_mean, device=device)
    m_model.bn1.running_var = t2m_param(t_model.bn1.running_var, device=device)

    batch_size = config[0]
    m_model.train_mode()
    t_model.train()
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)

    for i in range(batch_size):
        m_dy, t_dy = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False)
        m_x, t_x = randn_torch([1, 3, config[1], config[1]], requires_grad=True, device=device)
        m_y, t_y = one_hot_torch(size=1, num_classes=config[2], device=device)
        m_loss = run_vm_model(m_optimizer, device, [m_dy, m_x, m_y])
        t_optimizer.zero_grad()
        t_loss = t_model(t_x, t_y)
        t_loss.backward(t_dy)
        t_optimizer.step()
        check(m_model.conv1.w, t_model.conv1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.w, t_model.linear1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.linear1.b, t_model.linear1.bias, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.w, t_model.bn1.weight, rtol=1e-4, atol=1e-4)
        check(m_model.bn1.b, t_model.bn1.bias, rtol=1e-4, atol=1e-4)


@with_seed(0)
@pytest.mark.parametrize("device", get_testable_devices())
def test_mxnet_model(device):
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(64, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize(mx.init.Xavier(magnitude=2.24))
    mx_trainer = gluon.Trainer(
        net.collect_params(), "sgd", {"learning_rate": 0.1, "momentum": 0.01}
    )
    x, mx_x = randn_mxnet((1, 3, 224, 224), requires_grad=True, device=device)
    dy, mx_dy = randn_mxnet((1, 10), device=device)
    net(mx_x)
    model = raf.frontend.from_mxnet(net, ["x"])
    model.train_mode()
    model.to(device=device)
    trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(model)
    with mx.autograd.record():
        mx_loss = net(mx_x)
    mx_loss.backward(mx_dy)
    mx_trainer.step(1)
    loss = run_vm_model(trainer, device, [dy, x])[0]
    check(loss, mx_loss, rtol=1e-4, atol=1e-4)
    params = model.state()
    for name, param in net.collect_params().items():
        check(params[name], param.data(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@patch("raf.distributed.get_communicator")
@patch("raf.distributed.get_config")
def test_state_partition(mock_get_config, mock_get_comm):
    """Note that this test only verifies the IR with SGD without checking the correctness.
    Accordingly, this test does not require multiple devices.
    """
    # pylint: disable=too-many-locals, protected-access
    # Mock dist config & communicator to let with_lans generate the desired IR.
    class MockConfig:
        def __init__(self):
            self.enable_data_parallel = True
            self.zero_opt_level = 2
            self.group_bucket_size = 50000000

    mock_get_config.return_value = MockConfig()

    class MockComm:
        def __init__(self):
            self.size = 4
            self.rank = 3

    mock_get_comm.return_value = MockComm()

    shape, n_classes = 28, 10
    batch_size = 7
    m_model = RAFTest(shape, 10)
    m_model.train_mode()
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)

    device = "cuda"
    m_x, _ = randn_torch([batch_size, 3, shape, shape], requires_grad=True, device=device)
    m_dy, _ = randn_torch((), std=0.0, mean=1.0, device=device, requires_grad=False)
    m_ytrue, _ = one_hot_torch(size=batch_size, num_classes=n_classes, device=device)
    args = [m_dy, m_x, m_ytrue]

    record = m_optimizer._internal(*args)
    mod = record.mod
    text = raf.ir.AsText(mod)

    # Verify main function arguments.
    func_def = [line for line in text.split("\n") if line.startswith("def @main")]
    assert len(func_def) == 1
    # Extract all tensor arguments and create a {name -> first axis shape} map.
    param_map = {}
    for name, ttype in re.findall(r"%([^:]+): Tensor\[([^\]]+)\]", func_def[0]):
        param_map[name] = int(ttype[ttype.find("(") + 1 : ttype.find(")")].split(",")[0])
    for name, shape in param_map.items():
        if name.find("sgd_") == -1:
            continue
        param_name = f"model.{name[:-6]}"  # Find the original parameter.
        # The size of sgd status should be 1/4 of the original parameter.
        assert param_name in param_map and math.ceil(param_map[param_name] / 4) == shape

    # Verify IR. This model has 8 parameters and 10 gradients
    # (gradients for input data and ytrure are useless).
    # The 10 _reduce_scatters are grouped. So only 1 _group_reduce_scatter.
    assert text.count("raf.op._group_reduce_scatter") == 1, text
    assert text.count("raf.op._allgather") == 8, text
    assert text.count("raf.op.strided_slice") == 8, text


if __name__ == "__main__":
    pytest.main([__file__])
