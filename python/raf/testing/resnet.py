# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""resnet model for ImageNet"""
# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
import torch
import torch.nn as nn

import raf
from raf.model import BatchNorm, Conv2d, Linear, Sequential
from .common import check, randn_torch, t2m_param, one_hot_torch
from .utils import get_param, set_param


def _param_map(m_model, t_model):
    """maps from m_model param name to t_model param value, without params for shortcut"""
    assert m_model.num_blocks == t_model.num_blocks
    num_blocks = m_model.num_blocks
    m_bottleneck_names = [
        "conv1.w",
        "bn1.b",
        "bn1.w",
        "bn1.running_mean",
        "bn1.running_var",
        "conv2.w",
        "bn2.b",
        "bn2.w",
        "bn2.running_mean",
        "bn2.running_var",
        "conv3.w",
        "bn3.b",
        "bn3.w",
        "bn3.running_mean",
        "bn3.running_var",
    ]
    t_bottleneck_names = [
        "conv1.weight",
        "bn1.bias",
        "bn1.weight",
        "bn1.running_mean",
        "bn1.running_var",
        "conv2.weight",
        "bn2.bias",
        "bn2.weight",
        "bn2.running_mean",
        "bn2.running_var",
        "conv3.weight",
        "bn3.bias",
        "bn3.weight",
        "bn3.running_mean",
        "bn3.running_var",
    ]
    m_downsample_names = ["w", "b", "running_mean", "running_var"]
    t_downsample_names = ["weight", "bias", "running_mean", "running_var"]
    names = []
    values = []
    for i, num in enumerate(num_blocks):
        for j in range(num):
            m_names, t_names = m_bottleneck_names, t_bottleneck_names
            prefix = f"layer{i + 1}.seq_{j}"
            cur = [prefix + "." + name for name in m_names]
            names.extend(cur)
            layer = getattr(t_model, f"layer{i + 1}")
            layer = layer[j]
            cur = [get_param(layer, name) for name in t_names]
            values.extend(cur)
        m_names, t_names = m_downsample_names, t_downsample_names
        prefix = f"layer{i + 1}.seq_0.downsample"
        names.append(prefix + ".seq_0.w")
        cur = [prefix + ".seq_1." + name for name in m_names]
        names.extend(cur)
        layer = getattr(t_model, f"layer{i + 1}")
        layer = layer[0]
        downsample0 = getattr(layer, "downsample")[0]
        downsample1 = getattr(layer, "downsample")[1]
        values.append(downsample0.weight)
        cur = [getattr(downsample1, name) for name in t_names]
        values.extend(cur)
    return dict(zip(names, values))


def _init(m_model, t_model, device, params):
    """initialize raf model with parameters of torch model"""
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in params(m_model, t_model).items():
        set_param(m_model, m_name, t2m_param(t_w, device=device))


def _check_params(m_model, t_model, atol, rtol, params):
    """check the parameters of m_model and t_model"""
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in params(m_model, t_model).items():
        m_w = get_param(m_model, m_name)
        check(m_w, t_w, atol=atol, rtol=rtol)


def param_map(m_model, t_model):
    """maps from m_model parameter name to t_model parameter value"""
    # pylint: disable=too-many-locals
    assert m_model.num_blocks == t_model.num_blocks
    res = {
        "conv1.w": t_model.conv1.weight,
        "bn1.w": t_model.bn1.weight,
        "bn1.b": t_model.bn1.bias,
        "bn1.running_mean": t_model.bn1.running_mean,
        "bn1.running_var": t_model.bn1.running_var,
        "fc1.w": t_model.fc1.weight,
        "fc1.b": t_model.fc1.bias,
    }
    res.update(_param_map(m_model, t_model))
    return res


def init(m_model, t_model, device="cpu"):
    """initialize raf model with parameters of torch model"""
    _init(m_model, t_model, device, param_map)


def check_params(m_model, t_model, atol=1e-3, rtol=1e-3):
    """check the parameters of m_model and t_model"""
    _check_params(m_model, t_model, atol, rtol, param_map)


class TorchBottleneck(nn.Module):
    # pylint: disable=missing-function-docstring, abstract-method
    """torch BottleNeck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride):
        super(TorchBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
            groups=1,
            dilation=1,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or inplanes != planes * TorchBottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * TorchBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * TorchBottleneck.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class TorchResNet50(nn.Module):
    """torch ResNet50"""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method, missing-function-docstring
    def __init__(self, num_blocks, num_classes=1000):
        super(TorchResNet50, self).__init__()
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.fc1 = nn.Linear(512 * TorchBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        layers = [TorchBottleneck(self.inplanes, planes, stride)]
        self.inplanes = planes * TorchBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(TorchBottleneck(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward_infer(self, x):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.fc1(x)
        return x

    def forward(self, x, y_true):
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        y = self.forward_infer(x)
        y_pred = F.log_softmax(y, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


class RAFBottleneck(raf.Model):
    """raf BottleNeck"""

    # pylint: disable=missing-function-docstring
    expansion = 4

    def build(self, inplanes, planes, stride):
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
            groups=1,
            dilation=1,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        if stride != 1 or inplanes != planes * RAFBottleneck.expansion:
            self.downsample = Sequential(
                Conv2d(
                    inplanes,
                    planes * RAFBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * RAFBottleneck.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = raf.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = raf.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = raf.add(out, identity)
        out = raf.relu(out)
        return out


class RAFResNet50(raf.Model):
    """raf ResNet50"""

    # pylint: disable=missing-function-docstring, too-many-instance-attributes

    def build(self, num_blocks, num_classes=1000):
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(self.inplanes)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.fc1 = Linear(512 * RAFBottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [RAFBottleneck(self.inplanes, planes, stride)]
        self.inplanes = planes * RAFBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(RAFBottleneck(self.inplanes, planes, stride=1))
        return Sequential(*layers)

    @raf.model.trace
    def forward_infer(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = raf.relu(x)
        x = raf.max_pool2d(x, kernel=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = raf.avg_pool2d(x, kernel=7, stride=7)
        x = raf.batch_flatten(x)
        x = self.fc1(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss


def get_model(num_blocks, train=True):
    """get resnet model"""
    m_model = RAFResNet50(num_blocks)
    t_model = TorchResNet50(num_blocks)
    init(m_model, t_model)
    if train:
        m_model.train_mode()
        t_model.train()
    else:
        m_model.infer_mode()
        t_model.eval()
    return m_model, t_model


def get_input(batch_size=1, device="cuda", train=True):
    """get resnet input"""
    m_x, t_x = randn_torch([batch_size, 3, 224, 224], device=device, requires_grad=True)
    if not train:
        return [(m_x,), (t_x,)]
    m_y, t_y = one_hot_torch(batch_size, num_classes=1000, device=device)
    return [(m_x, m_y), (t_x, t_y)]
