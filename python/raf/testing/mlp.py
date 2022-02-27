# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP model"""
# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
# pylint: disable=missing-class-docstring, too-many-arguments, missing-function-docstring
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf.model import Linear
from .common import check, randn_torch, t2m_param, one_hot_torch
from .utils import get_param, set_param


class TorchMlp(nn.Module):  # pylint: disable=abstract-method
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(TorchMlp, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, num_outputs)

    def forward_infer(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return y

    def forward(self, x, y_true=None):  # pylint: disable=arguments-differ
        y = self.forward_infer(x)
        if self.training:
            y_pred = F.log_softmax(y, dim=-1)
            loss = F.nll_loss(y_pred, y_true)
            return loss
        return y


class RAFMlp(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        self.fc1 = Linear(num_inputs, num_hiddens1)
        self.fc2 = Linear(num_hiddens1, num_hiddens2)
        self.fc3 = Linear(num_hiddens2, num_outputs)

    @raf.model.trace
    def forward_infer(self, x):
        y = self.fc1(x)
        y = raf.relu(y)
        y = self.fc2(y)
        y = raf.relu(y)
        y = self.fc3(y)
        return y

    @raf.model.trace
    def forward(self, x, y_true):
        y = self.forward_infer(x)
        y_pred = raf.log_softmax(y)
        loss = raf.nll_loss(y_true, y_pred)
        return loss


def _param_map(t_model):
    """maps from m_model parameter name to t_model parameter value"""
    res = {
        "fc1.w": t_model.fc1.weight,
        "fc1.b": t_model.fc1.bias,
        "fc2.w": t_model.fc2.weight,
        "fc2.b": t_model.fc2.bias,
        "fc3.w": t_model.fc3.weight,
        "fc3.b": t_model.fc3.bias,
    }
    return res


def _init(m_model, t_model, device="cpu"):
    """initialize meta model with parameters of torch model"""
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in _param_map(t_model).items():
        set_param(m_model, m_name, t2m_param(t_w, device=device))


def check_params(m_model, t_model, atol=1e-4, rtol=1e-4):
    """check the parameters of m_model and t_model"""
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in _param_map(t_model).items():
        m_w = get_param(m_model, m_name)
        check(m_w, t_w, atol=atol, rtol=rtol)


def get_model(config, train=True):
    """get MLP model"""
    m_model = RAFMlp(*config)
    t_model = TorchMlp(*config)
    _init(m_model, t_model)
    if train:
        m_model.train_mode()
        t_model.train()
    else:
        m_model.infer_mode()
        t_model.eval()
    return m_model, t_model


def get_input(config, batch_size=1, device="cpu", train=True):
    """get MLP input"""
    m_x, t_x = randn_torch([batch_size, config[0]], device=device, requires_grad=True)
    if not train:
        return [(m_x,), (t_x,)]
    m_y, t_y = one_hot_torch(batch_size, num_classes=config[1], device=device)
    return [(m_x, m_y), (t_x, t_y)]
