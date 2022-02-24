# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf.model import Linear
from raf.testing import check, one_hot_torch, randn_torch, t2m_param


class TorchMlp(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(TorchMlp, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, num_outputs)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class RAFMlp(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        self.fc1 = Linear(num_inputs, num_hiddens1)
        self.fc2 = Linear(num_hiddens1, num_hiddens2)
        self.fc3 = Linear(num_hiddens2, num_outputs)

    @raf.model.trace
    def forward(self, x):
        x = self.fc1(x)
        x = raf.relu(x)
        x = self.fc2(x)
        x = raf.relu(x)
        x = self.fc3(x)
        return x


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "config",
    [
        (784, 10, 256, 256),
        (512, 64, 128, 128),
        (4, 2, 3, 3),
    ],
)
@pytest.mark.parametrize("is_train", [True, False])
def test_mlp(config, is_train):
    m_model = RAFMlp(*config)
    m_model.to(device="cuda")
    t_model = TorchMlp(*config)
    t_model.to(device="cuda")
    m_model.fc1.w = t2m_param(t_model.fc1.weight)
    m_model.fc1.b = t2m_param(t_model.fc1.bias)
    m_model.fc2.w = t2m_param(t_model.fc2.weight)
    m_model.fc2.b = t2m_param(t_model.fc2.bias)
    m_model.fc3.w = t2m_param(t_model.fc3.weight)
    m_model.fc3.b = t2m_param(t_model.fc3.bias)

    m_x, t_x = randn_torch((1, config[0]), requires_grad=is_train, device="cuda")
    m_y, t_y = one_hot_torch(batch_size=1, num_classes=config[-1])
    if is_train:
        m_model.train_mode()
        t_model.train()
    else:
        m_model.infer_mode()
        t_model.eval()
    m_y = m_model(m_x)
    t_y = t_model(t_x)
    if is_train:
        m_dy, t_dy = randn_torch(m_y.shape, std=m_y.numpy().std() * 0.0001, device="cuda")
        t_y.backward(t_dy)
        m_y.backward(m_dy)
        check(m_model.fc1.w.grad, t_model.fc1.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.fc1.b.grad, t_model.fc1.bias.grad, rtol=1e-4, atol=1e-4)
        check(m_model.fc2.w.grad, t_model.fc2.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.fc2.b.grad, t_model.fc2.bias.grad, rtol=1e-4, atol=1e-4)
        check(m_model.fc3.w.grad, t_model.fc3.weight.grad, rtol=1e-4, atol=1e-4)
        check(m_model.fc3.b.grad, t_model.fc3.bias.grad, rtol=1e-4, atol=1e-4)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
