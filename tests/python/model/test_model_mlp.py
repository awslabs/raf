import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mnm
from mnm.model import Linear


def t2m_param(param):
    return mnm.ndarray(param.detach().numpy(), ctx="cuda")  # pylint: disable=unexpected-keyword-arg


def one_hot(batch_size, num_classes, ctx="cuda", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    t_x = torch.tensor(targets, requires_grad=False)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


def randn(shape, *, ctx="cuda", dtype="float32", std=1.0, mean=0.0, requires_grad=False):
    x = np.random.randn(*shape) * std + mean
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    if requires_grad:
        m_x.requires_grad = True
    t_x = torch.tensor(x, requires_grad=requires_grad)  # pylint: disable=not-callable
    return m_x, t_x


class TorchMlp(nn.Module):
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


class MNMMlp(mnm.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        self.fc1 = Linear(num_inputs, num_hiddens1)
        self.fc2 = Linear(num_hiddens1, num_hiddens2)
        self.fc3 = Linear(num_hiddens2, num_outputs)

    @mnm.model.trace
    def forward(self, x):
        x = self.fc1(x)
        x = mnm.relu(x)
        x = self.fc2(x)
        x = mnm.relu(x)
        x = self.fc3(x)
        return x


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("config", [
    (784, 10, 256, 256),
    (512, 64, 128, 128),
    (4, 2, 3, 3),
])
@pytest.mark.parametrize("is_train", [True, False])
def test_mlp(config, is_train):
    m_model = MNMMlp(*config)
    t_model = TorchMlp(*config)
    m_model.fc1.w = t2m_param(t_model.fc1.weight)
    m_model.fc1.b = t2m_param(t_model.fc1.bias)
    m_model.fc2.w = t2m_param(t_model.fc2.weight)
    m_model.fc2.b = t2m_param(t_model.fc2.bias)
    m_model.fc3.w = t2m_param(t_model.fc3.weight)
    m_model.fc3.b = t2m_param(t_model.fc3.bias)

    m_x, t_x = randn((1, config[0]), requires_grad=is_train)
    m_y, t_y = one_hot(batch_size=1, num_classes=config[-1])
    if is_train:
        m_model.train_mode()
        t_model.train()
    else:
        m_model.infer_mode()
        t_model.eval()
    m_y = m_model(m_x)
    t_y = t_model(t_x)
    if is_train:
        m_dy, t_dy = randn(m_y.shape, std=m_y.asnumpy().std() * 0.0001)
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
