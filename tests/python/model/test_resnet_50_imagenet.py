# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines
import pytest
import torch
import torch.nn as nn

import raf
from raf.model import BatchNorm, Conv2d, Linear, Sequential
from raf.testing import check, one_hot_torch, randn_torch, t2m_param


class TorchBottleneck(nn.Module):  # pylint: disable=abstract-method
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
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method
    def __init__(self, layers, num_classes=1000):
        super(TorchResNet50, self).__init__()
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
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
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

    def forward(self, x, y_true):  # pylint: disable=arguments-differ
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
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


class RAFBottleneck(raf.Model):
    expansion = 4

    # pylint: disable=attribute-defined-outside-init
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

    # pylint: enable=attribute-defined-outside-init

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


class RAFResNet50(raf.Model):  # pylint: disable=too-many-instance-attributes

    # pylint: disable=attribute-defined-outside-init

    def build(self, layers, num_classes=1000):
        self.inplanes = 64
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.fc1 = Linear(512 * RAFBottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [RAFBottleneck(self.inplanes, planes, stride)]
        self.inplanes = planes * RAFBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(RAFBottleneck(self.inplanes, planes, stride=1))
        return Sequential(*layers)

    # pylint: enable=attribute-defined-outside-init

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


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_r50_v1_imagenet():  # pylint: disable=too-many-statements
    t_model = TorchResNet50([3, 4, 6, 3])
    t_model.to(device="cuda")
    m_model = RAFResNet50([3, 4, 6, 3])
    # pylint: disable=no-member,line-too-long
    m_model.conv1.w = t2m_param(t_model.conv1.weight)
    m_model.bn1.w = t2m_param(t_model.bn1.weight)
    m_model.bn1.b = t2m_param(t_model.bn1.bias)
    m_model.bn1.running_mean = t2m_param(t_model.bn1.running_mean)
    m_model.bn1.running_var = t2m_param(t_model.bn1.running_var)
    m_model.layer1.seq_0.conv1.w = t2m_param(t_model.layer1[0].conv1.weight)
    m_model.layer1.seq_0.bn1.w = t2m_param(t_model.layer1[0].bn1.weight)
    m_model.layer1.seq_0.bn1.b = t2m_param(t_model.layer1[0].bn1.bias)
    m_model.layer1.seq_0.bn1.running_mean = t2m_param(t_model.layer1[0].bn1.running_mean)
    m_model.layer1.seq_0.bn1.running_var = t2m_param(t_model.layer1[0].bn1.running_var)
    m_model.layer1.seq_0.conv2.w = t2m_param(t_model.layer1[0].conv2.weight)
    m_model.layer1.seq_0.bn2.w = t2m_param(t_model.layer1[0].bn2.weight)
    m_model.layer1.seq_0.bn2.b = t2m_param(t_model.layer1[0].bn2.bias)
    m_model.layer1.seq_0.bn2.running_mean = t2m_param(t_model.layer1[0].bn2.running_mean)
    m_model.layer1.seq_0.bn2.running_var = t2m_param(t_model.layer1[0].bn2.running_var)
    m_model.layer1.seq_0.conv3.w = t2m_param(t_model.layer1[0].conv3.weight)
    m_model.layer1.seq_0.bn3.w = t2m_param(t_model.layer1[0].bn3.weight)
    m_model.layer1.seq_0.bn3.b = t2m_param(t_model.layer1[0].bn3.bias)
    m_model.layer1.seq_0.bn3.running_mean = t2m_param(t_model.layer1[0].bn3.running_mean)
    m_model.layer1.seq_0.bn3.running_var = t2m_param(t_model.layer1[0].bn3.running_var)
    m_model.layer1.seq_0.downsample.seq_0.w = t2m_param(t_model.layer1[0].downsample[0].weight)
    m_model.layer1.seq_0.downsample.seq_1.w = t2m_param(t_model.layer1[0].downsample[1].weight)
    m_model.layer1.seq_0.downsample.seq_1.b = t2m_param(t_model.layer1[0].downsample[1].bias)
    m_model.layer1.seq_0.downsample.seq_1.running_mean = t2m_param(
        t_model.layer1[0].downsample[1].running_mean
    )
    m_model.layer1.seq_0.downsample.seq_1.running_var = t2m_param(
        t_model.layer1[0].downsample[1].running_var
    )
    m_model.layer1.seq_1.conv1.w = t2m_param(t_model.layer1[1].conv1.weight)
    m_model.layer1.seq_1.bn1.w = t2m_param(t_model.layer1[1].bn1.weight)
    m_model.layer1.seq_1.bn1.b = t2m_param(t_model.layer1[1].bn1.bias)
    m_model.layer1.seq_1.bn1.running_mean = t2m_param(t_model.layer1[1].bn1.running_mean)
    m_model.layer1.seq_1.bn1.running_var = t2m_param(t_model.layer1[1].bn1.running_var)
    m_model.layer1.seq_1.conv2.w = t2m_param(t_model.layer1[1].conv2.weight)
    m_model.layer1.seq_1.bn2.w = t2m_param(t_model.layer1[1].bn2.weight)
    m_model.layer1.seq_1.bn2.b = t2m_param(t_model.layer1[1].bn2.bias)
    m_model.layer1.seq_1.bn2.running_mean = t2m_param(t_model.layer1[1].bn2.running_mean)
    m_model.layer1.seq_1.bn2.running_var = t2m_param(t_model.layer1[1].bn2.running_var)
    m_model.layer1.seq_1.conv3.w = t2m_param(t_model.layer1[1].conv3.weight)
    m_model.layer1.seq_1.bn3.w = t2m_param(t_model.layer1[1].bn3.weight)
    m_model.layer1.seq_1.bn3.b = t2m_param(t_model.layer1[1].bn3.bias)
    m_model.layer1.seq_1.bn3.running_mean = t2m_param(t_model.layer1[1].bn3.running_mean)
    m_model.layer1.seq_1.bn3.running_var = t2m_param(t_model.layer1[1].bn3.running_var)
    m_model.layer1.seq_2.conv1.w = t2m_param(t_model.layer1[2].conv1.weight)
    m_model.layer1.seq_2.bn1.w = t2m_param(t_model.layer1[2].bn1.weight)
    m_model.layer1.seq_2.bn1.b = t2m_param(t_model.layer1[2].bn1.bias)
    m_model.layer1.seq_2.bn1.running_mean = t2m_param(t_model.layer1[2].bn1.running_mean)
    m_model.layer1.seq_2.bn1.running_var = t2m_param(t_model.layer1[2].bn1.running_var)
    m_model.layer1.seq_2.conv2.w = t2m_param(t_model.layer1[2].conv2.weight)
    m_model.layer1.seq_2.bn2.w = t2m_param(t_model.layer1[2].bn2.weight)
    m_model.layer1.seq_2.bn2.b = t2m_param(t_model.layer1[2].bn2.bias)
    m_model.layer1.seq_2.bn2.running_mean = t2m_param(t_model.layer1[2].bn2.running_mean)
    m_model.layer1.seq_2.bn2.running_var = t2m_param(t_model.layer1[2].bn2.running_var)
    m_model.layer1.seq_2.conv3.w = t2m_param(t_model.layer1[2].conv3.weight)
    m_model.layer1.seq_2.bn3.w = t2m_param(t_model.layer1[2].bn3.weight)
    m_model.layer1.seq_2.bn3.b = t2m_param(t_model.layer1[2].bn3.bias)
    m_model.layer1.seq_2.bn3.running_mean = t2m_param(t_model.layer1[2].bn3.running_mean)
    m_model.layer1.seq_2.bn3.running_var = t2m_param(t_model.layer1[2].bn3.running_var)
    m_model.layer2.seq_0.conv1.w = t2m_param(t_model.layer2[0].conv1.weight)
    m_model.layer2.seq_0.bn1.w = t2m_param(t_model.layer2[0].bn1.weight)
    m_model.layer2.seq_0.bn1.b = t2m_param(t_model.layer2[0].bn1.bias)
    m_model.layer2.seq_0.bn1.running_mean = t2m_param(t_model.layer2[0].bn1.running_mean)
    m_model.layer2.seq_0.bn1.running_var = t2m_param(t_model.layer2[0].bn1.running_var)
    m_model.layer2.seq_0.conv2.w = t2m_param(t_model.layer2[0].conv2.weight)
    m_model.layer2.seq_0.bn2.w = t2m_param(t_model.layer2[0].bn2.weight)
    m_model.layer2.seq_0.bn2.b = t2m_param(t_model.layer2[0].bn2.bias)
    m_model.layer2.seq_0.bn2.running_mean = t2m_param(t_model.layer2[0].bn2.running_mean)
    m_model.layer2.seq_0.bn2.running_var = t2m_param(t_model.layer2[0].bn2.running_var)
    m_model.layer2.seq_0.conv3.w = t2m_param(t_model.layer2[0].conv3.weight)
    m_model.layer2.seq_0.bn3.w = t2m_param(t_model.layer2[0].bn3.weight)
    m_model.layer2.seq_0.bn3.b = t2m_param(t_model.layer2[0].bn3.bias)
    m_model.layer2.seq_0.bn3.running_mean = t2m_param(t_model.layer2[0].bn3.running_mean)
    m_model.layer2.seq_0.bn3.running_var = t2m_param(t_model.layer2[0].bn3.running_var)
    m_model.layer2.seq_0.downsample.seq_0.w = t2m_param(t_model.layer2[0].downsample[0].weight)
    m_model.layer2.seq_0.downsample.seq_1.w = t2m_param(t_model.layer2[0].downsample[1].weight)
    m_model.layer2.seq_0.downsample.seq_1.b = t2m_param(t_model.layer2[0].downsample[1].bias)
    m_model.layer2.seq_0.downsample.seq_1.running_mean = t2m_param(
        t_model.layer2[0].downsample[1].running_mean
    )
    m_model.layer2.seq_0.downsample.seq_1.running_var = t2m_param(
        t_model.layer2[0].downsample[1].running_var
    )
    m_model.layer2.seq_1.conv1.w = t2m_param(t_model.layer2[1].conv1.weight)
    m_model.layer2.seq_1.bn1.w = t2m_param(t_model.layer2[1].bn1.weight)
    m_model.layer2.seq_1.bn1.b = t2m_param(t_model.layer2[1].bn1.bias)
    m_model.layer2.seq_1.bn1.running_mean = t2m_param(t_model.layer2[1].bn1.running_mean)
    m_model.layer2.seq_1.bn1.running_var = t2m_param(t_model.layer2[1].bn1.running_var)
    m_model.layer2.seq_1.conv2.w = t2m_param(t_model.layer2[1].conv2.weight)
    m_model.layer2.seq_1.bn2.w = t2m_param(t_model.layer2[1].bn2.weight)
    m_model.layer2.seq_1.bn2.b = t2m_param(t_model.layer2[1].bn2.bias)
    m_model.layer2.seq_1.bn2.running_mean = t2m_param(t_model.layer2[1].bn2.running_mean)
    m_model.layer2.seq_1.bn2.running_var = t2m_param(t_model.layer2[1].bn2.running_var)
    m_model.layer2.seq_1.conv3.w = t2m_param(t_model.layer2[1].conv3.weight)
    m_model.layer2.seq_1.bn3.w = t2m_param(t_model.layer2[1].bn3.weight)
    m_model.layer2.seq_1.bn3.b = t2m_param(t_model.layer2[1].bn3.bias)
    m_model.layer2.seq_1.bn3.running_mean = t2m_param(t_model.layer2[1].bn3.running_mean)
    m_model.layer2.seq_1.bn3.running_var = t2m_param(t_model.layer2[1].bn3.running_var)
    m_model.layer2.seq_2.conv1.w = t2m_param(t_model.layer2[2].conv1.weight)
    m_model.layer2.seq_2.bn1.w = t2m_param(t_model.layer2[2].bn1.weight)
    m_model.layer2.seq_2.bn1.b = t2m_param(t_model.layer2[2].bn1.bias)
    m_model.layer2.seq_2.bn1.running_mean = t2m_param(t_model.layer2[2].bn1.running_mean)
    m_model.layer2.seq_2.bn1.running_var = t2m_param(t_model.layer2[2].bn1.running_var)
    m_model.layer2.seq_2.conv2.w = t2m_param(t_model.layer2[2].conv2.weight)
    m_model.layer2.seq_2.bn2.w = t2m_param(t_model.layer2[2].bn2.weight)
    m_model.layer2.seq_2.bn2.b = t2m_param(t_model.layer2[2].bn2.bias)
    m_model.layer2.seq_2.bn2.running_mean = t2m_param(t_model.layer2[2].bn2.running_mean)
    m_model.layer2.seq_2.bn2.running_var = t2m_param(t_model.layer2[2].bn2.running_var)
    m_model.layer2.seq_2.conv3.w = t2m_param(t_model.layer2[2].conv3.weight)
    m_model.layer2.seq_2.bn3.w = t2m_param(t_model.layer2[2].bn3.weight)
    m_model.layer2.seq_2.bn3.b = t2m_param(t_model.layer2[2].bn3.bias)
    m_model.layer2.seq_2.bn3.running_mean = t2m_param(t_model.layer2[2].bn3.running_mean)
    m_model.layer2.seq_2.bn3.running_var = t2m_param(t_model.layer2[2].bn3.running_var)
    m_model.layer2.seq_3.conv1.w = t2m_param(t_model.layer2[3].conv1.weight)
    m_model.layer2.seq_3.bn1.w = t2m_param(t_model.layer2[3].bn1.weight)
    m_model.layer2.seq_3.bn1.b = t2m_param(t_model.layer2[3].bn1.bias)
    m_model.layer2.seq_3.bn1.running_mean = t2m_param(t_model.layer2[3].bn1.running_mean)
    m_model.layer2.seq_3.bn1.running_var = t2m_param(t_model.layer2[3].bn1.running_var)
    m_model.layer2.seq_3.conv2.w = t2m_param(t_model.layer2[3].conv2.weight)
    m_model.layer2.seq_3.bn2.w = t2m_param(t_model.layer2[3].bn2.weight)
    m_model.layer2.seq_3.bn2.b = t2m_param(t_model.layer2[3].bn2.bias)
    m_model.layer2.seq_3.bn2.running_mean = t2m_param(t_model.layer2[3].bn2.running_mean)
    m_model.layer2.seq_3.bn2.running_var = t2m_param(t_model.layer2[3].bn2.running_var)
    m_model.layer2.seq_3.conv3.w = t2m_param(t_model.layer2[3].conv3.weight)
    m_model.layer2.seq_3.bn3.w = t2m_param(t_model.layer2[3].bn3.weight)
    m_model.layer2.seq_3.bn3.b = t2m_param(t_model.layer2[3].bn3.bias)
    m_model.layer2.seq_3.bn3.running_mean = t2m_param(t_model.layer2[3].bn3.running_mean)
    m_model.layer2.seq_3.bn3.running_var = t2m_param(t_model.layer2[3].bn3.running_var)
    m_model.layer3.seq_0.conv1.w = t2m_param(t_model.layer3[0].conv1.weight)
    m_model.layer3.seq_0.bn1.w = t2m_param(t_model.layer3[0].bn1.weight)
    m_model.layer3.seq_0.bn1.b = t2m_param(t_model.layer3[0].bn1.bias)
    m_model.layer3.seq_0.bn1.running_mean = t2m_param(t_model.layer3[0].bn1.running_mean)
    m_model.layer3.seq_0.bn1.running_var = t2m_param(t_model.layer3[0].bn1.running_var)
    m_model.layer3.seq_0.conv2.w = t2m_param(t_model.layer3[0].conv2.weight)
    m_model.layer3.seq_0.bn2.w = t2m_param(t_model.layer3[0].bn2.weight)
    m_model.layer3.seq_0.bn2.b = t2m_param(t_model.layer3[0].bn2.bias)
    m_model.layer3.seq_0.bn2.running_mean = t2m_param(t_model.layer3[0].bn2.running_mean)
    m_model.layer3.seq_0.bn2.running_var = t2m_param(t_model.layer3[0].bn2.running_var)
    m_model.layer3.seq_0.conv3.w = t2m_param(t_model.layer3[0].conv3.weight)
    m_model.layer3.seq_0.bn3.w = t2m_param(t_model.layer3[0].bn3.weight)
    m_model.layer3.seq_0.bn3.b = t2m_param(t_model.layer3[0].bn3.bias)
    m_model.layer3.seq_0.bn3.running_mean = t2m_param(t_model.layer3[0].bn3.running_mean)
    m_model.layer3.seq_0.bn3.running_var = t2m_param(t_model.layer3[0].bn3.running_var)
    m_model.layer3.seq_0.downsample.seq_0.w = t2m_param(t_model.layer3[0].downsample[0].weight)
    m_model.layer3.seq_0.downsample.seq_1.w = t2m_param(t_model.layer3[0].downsample[1].weight)
    m_model.layer3.seq_0.downsample.seq_1.b = t2m_param(t_model.layer3[0].downsample[1].bias)
    m_model.layer3.seq_0.downsample.seq_1.running_mean = t2m_param(
        t_model.layer3[0].downsample[1].running_mean
    )
    m_model.layer3.seq_0.downsample.seq_1.running_var = t2m_param(
        t_model.layer3[0].downsample[1].running_var
    )
    m_model.layer3.seq_1.conv1.w = t2m_param(t_model.layer3[1].conv1.weight)
    m_model.layer3.seq_1.bn1.w = t2m_param(t_model.layer3[1].bn1.weight)
    m_model.layer3.seq_1.bn1.b = t2m_param(t_model.layer3[1].bn1.bias)
    m_model.layer3.seq_1.bn1.running_mean = t2m_param(t_model.layer3[1].bn1.running_mean)
    m_model.layer3.seq_1.bn1.running_var = t2m_param(t_model.layer3[1].bn1.running_var)
    m_model.layer3.seq_1.conv2.w = t2m_param(t_model.layer3[1].conv2.weight)
    m_model.layer3.seq_1.bn2.w = t2m_param(t_model.layer3[1].bn2.weight)
    m_model.layer3.seq_1.bn2.b = t2m_param(t_model.layer3[1].bn2.bias)
    m_model.layer3.seq_1.bn2.running_mean = t2m_param(t_model.layer3[1].bn2.running_mean)
    m_model.layer3.seq_1.bn2.running_var = t2m_param(t_model.layer3[1].bn2.running_var)
    m_model.layer3.seq_1.conv3.w = t2m_param(t_model.layer3[1].conv3.weight)
    m_model.layer3.seq_1.bn3.w = t2m_param(t_model.layer3[1].bn3.weight)
    m_model.layer3.seq_1.bn3.b = t2m_param(t_model.layer3[1].bn3.bias)
    m_model.layer3.seq_1.bn3.running_mean = t2m_param(t_model.layer3[1].bn3.running_mean)
    m_model.layer3.seq_1.bn3.running_var = t2m_param(t_model.layer3[1].bn3.running_var)
    m_model.layer3.seq_2.conv1.w = t2m_param(t_model.layer3[2].conv1.weight)
    m_model.layer3.seq_2.bn1.w = t2m_param(t_model.layer3[2].bn1.weight)
    m_model.layer3.seq_2.bn1.b = t2m_param(t_model.layer3[2].bn1.bias)
    m_model.layer3.seq_2.bn1.running_mean = t2m_param(t_model.layer3[2].bn1.running_mean)
    m_model.layer3.seq_2.bn1.running_var = t2m_param(t_model.layer3[2].bn1.running_var)
    m_model.layer3.seq_2.conv2.w = t2m_param(t_model.layer3[2].conv2.weight)
    m_model.layer3.seq_2.bn2.w = t2m_param(t_model.layer3[2].bn2.weight)
    m_model.layer3.seq_2.bn2.b = t2m_param(t_model.layer3[2].bn2.bias)
    m_model.layer3.seq_2.bn2.running_mean = t2m_param(t_model.layer3[2].bn2.running_mean)
    m_model.layer3.seq_2.bn2.running_var = t2m_param(t_model.layer3[2].bn2.running_var)
    m_model.layer3.seq_2.conv3.w = t2m_param(t_model.layer3[2].conv3.weight)
    m_model.layer3.seq_2.bn3.w = t2m_param(t_model.layer3[2].bn3.weight)
    m_model.layer3.seq_2.bn3.b = t2m_param(t_model.layer3[2].bn3.bias)
    m_model.layer3.seq_2.bn3.running_mean = t2m_param(t_model.layer3[2].bn3.running_mean)
    m_model.layer3.seq_2.bn3.running_var = t2m_param(t_model.layer3[2].bn3.running_var)
    m_model.layer3.seq_3.conv1.w = t2m_param(t_model.layer3[3].conv1.weight)
    m_model.layer3.seq_3.bn1.w = t2m_param(t_model.layer3[3].bn1.weight)
    m_model.layer3.seq_3.bn1.b = t2m_param(t_model.layer3[3].bn1.bias)
    m_model.layer3.seq_3.bn1.running_mean = t2m_param(t_model.layer3[3].bn1.running_mean)
    m_model.layer3.seq_3.bn1.running_var = t2m_param(t_model.layer3[3].bn1.running_var)
    m_model.layer3.seq_3.conv2.w = t2m_param(t_model.layer3[3].conv2.weight)
    m_model.layer3.seq_3.bn2.w = t2m_param(t_model.layer3[3].bn2.weight)
    m_model.layer3.seq_3.bn2.b = t2m_param(t_model.layer3[3].bn2.bias)
    m_model.layer3.seq_3.bn2.running_mean = t2m_param(t_model.layer3[3].bn2.running_mean)
    m_model.layer3.seq_3.bn2.running_var = t2m_param(t_model.layer3[3].bn2.running_var)
    m_model.layer3.seq_3.conv3.w = t2m_param(t_model.layer3[3].conv3.weight)
    m_model.layer3.seq_3.bn3.w = t2m_param(t_model.layer3[3].bn3.weight)
    m_model.layer3.seq_3.bn3.b = t2m_param(t_model.layer3[3].bn3.bias)
    m_model.layer3.seq_3.bn3.running_mean = t2m_param(t_model.layer3[3].bn3.running_mean)
    m_model.layer3.seq_3.bn3.running_var = t2m_param(t_model.layer3[3].bn3.running_var)
    m_model.layer3.seq_4.conv1.w = t2m_param(t_model.layer3[4].conv1.weight)
    m_model.layer3.seq_4.bn1.w = t2m_param(t_model.layer3[4].bn1.weight)
    m_model.layer3.seq_4.bn1.b = t2m_param(t_model.layer3[4].bn1.bias)
    m_model.layer3.seq_4.bn1.running_mean = t2m_param(t_model.layer3[4].bn1.running_mean)
    m_model.layer3.seq_4.bn1.running_var = t2m_param(t_model.layer3[4].bn1.running_var)
    m_model.layer3.seq_4.conv2.w = t2m_param(t_model.layer3[4].conv2.weight)
    m_model.layer3.seq_4.bn2.w = t2m_param(t_model.layer3[4].bn2.weight)
    m_model.layer3.seq_4.bn2.b = t2m_param(t_model.layer3[4].bn2.bias)
    m_model.layer3.seq_4.bn2.running_mean = t2m_param(t_model.layer3[4].bn2.running_mean)
    m_model.layer3.seq_4.bn2.running_var = t2m_param(t_model.layer3[4].bn2.running_var)
    m_model.layer3.seq_4.conv3.w = t2m_param(t_model.layer3[4].conv3.weight)
    m_model.layer3.seq_4.bn3.w = t2m_param(t_model.layer3[4].bn3.weight)
    m_model.layer3.seq_4.bn3.b = t2m_param(t_model.layer3[4].bn3.bias)
    m_model.layer3.seq_4.bn3.running_mean = t2m_param(t_model.layer3[4].bn3.running_mean)
    m_model.layer3.seq_4.bn3.running_var = t2m_param(t_model.layer3[4].bn3.running_var)
    m_model.layer3.seq_5.conv1.w = t2m_param(t_model.layer3[5].conv1.weight)
    m_model.layer3.seq_5.bn1.w = t2m_param(t_model.layer3[5].bn1.weight)
    m_model.layer3.seq_5.bn1.b = t2m_param(t_model.layer3[5].bn1.bias)
    m_model.layer3.seq_5.bn1.running_mean = t2m_param(t_model.layer3[5].bn1.running_mean)
    m_model.layer3.seq_5.bn1.running_var = t2m_param(t_model.layer3[5].bn1.running_var)
    m_model.layer3.seq_5.conv2.w = t2m_param(t_model.layer3[5].conv2.weight)
    m_model.layer3.seq_5.bn2.w = t2m_param(t_model.layer3[5].bn2.weight)
    m_model.layer3.seq_5.bn2.b = t2m_param(t_model.layer3[5].bn2.bias)
    m_model.layer3.seq_5.bn2.running_mean = t2m_param(t_model.layer3[5].bn2.running_mean)
    m_model.layer3.seq_5.bn2.running_var = t2m_param(t_model.layer3[5].bn2.running_var)
    m_model.layer3.seq_5.conv3.w = t2m_param(t_model.layer3[5].conv3.weight)
    m_model.layer3.seq_5.bn3.w = t2m_param(t_model.layer3[5].bn3.weight)
    m_model.layer3.seq_5.bn3.b = t2m_param(t_model.layer3[5].bn3.bias)
    m_model.layer3.seq_5.bn3.running_mean = t2m_param(t_model.layer3[5].bn3.running_mean)
    m_model.layer3.seq_5.bn3.running_var = t2m_param(t_model.layer3[5].bn3.running_var)
    m_model.layer4.seq_0.conv1.w = t2m_param(t_model.layer4[0].conv1.weight)
    m_model.layer4.seq_0.bn1.w = t2m_param(t_model.layer4[0].bn1.weight)
    m_model.layer4.seq_0.bn1.b = t2m_param(t_model.layer4[0].bn1.bias)
    m_model.layer4.seq_0.bn1.running_mean = t2m_param(t_model.layer4[0].bn1.running_mean)
    m_model.layer4.seq_0.bn1.running_var = t2m_param(t_model.layer4[0].bn1.running_var)
    m_model.layer4.seq_0.conv2.w = t2m_param(t_model.layer4[0].conv2.weight)
    m_model.layer4.seq_0.bn2.w = t2m_param(t_model.layer4[0].bn2.weight)
    m_model.layer4.seq_0.bn2.b = t2m_param(t_model.layer4[0].bn2.bias)
    m_model.layer4.seq_0.bn2.running_mean = t2m_param(t_model.layer4[0].bn2.running_mean)
    m_model.layer4.seq_0.bn2.running_var = t2m_param(t_model.layer4[0].bn2.running_var)
    m_model.layer4.seq_0.conv3.w = t2m_param(t_model.layer4[0].conv3.weight)
    m_model.layer4.seq_0.bn3.w = t2m_param(t_model.layer4[0].bn3.weight)
    m_model.layer4.seq_0.bn3.b = t2m_param(t_model.layer4[0].bn3.bias)
    m_model.layer4.seq_0.bn3.running_mean = t2m_param(t_model.layer4[0].bn3.running_mean)
    m_model.layer4.seq_0.bn3.running_var = t2m_param(t_model.layer4[0].bn3.running_var)
    m_model.layer4.seq_0.downsample.seq_0.w = t2m_param(t_model.layer4[0].downsample[0].weight)
    m_model.layer4.seq_0.downsample.seq_1.w = t2m_param(t_model.layer4[0].downsample[1].weight)
    m_model.layer4.seq_0.downsample.seq_1.b = t2m_param(t_model.layer4[0].downsample[1].bias)
    m_model.layer4.seq_0.downsample.seq_1.running_mean = t2m_param(
        t_model.layer4[0].downsample[1].running_mean
    )
    m_model.layer4.seq_0.downsample.seq_1.running_var = t2m_param(
        t_model.layer4[0].downsample[1].running_var
    )
    m_model.layer4.seq_1.conv1.w = t2m_param(t_model.layer4[1].conv1.weight)
    m_model.layer4.seq_1.bn1.w = t2m_param(t_model.layer4[1].bn1.weight)
    m_model.layer4.seq_1.bn1.b = t2m_param(t_model.layer4[1].bn1.bias)
    m_model.layer4.seq_1.bn1.running_mean = t2m_param(t_model.layer4[1].bn1.running_mean)
    m_model.layer4.seq_1.bn1.running_var = t2m_param(t_model.layer4[1].bn1.running_var)
    m_model.layer4.seq_1.conv2.w = t2m_param(t_model.layer4[1].conv2.weight)
    m_model.layer4.seq_1.bn2.w = t2m_param(t_model.layer4[1].bn2.weight)
    m_model.layer4.seq_1.bn2.b = t2m_param(t_model.layer4[1].bn2.bias)
    m_model.layer4.seq_1.bn2.running_mean = t2m_param(t_model.layer4[1].bn2.running_mean)
    m_model.layer4.seq_1.bn2.running_var = t2m_param(t_model.layer4[1].bn2.running_var)
    m_model.layer4.seq_1.conv3.w = t2m_param(t_model.layer4[1].conv3.weight)
    m_model.layer4.seq_1.bn3.w = t2m_param(t_model.layer4[1].bn3.weight)
    m_model.layer4.seq_1.bn3.b = t2m_param(t_model.layer4[1].bn3.bias)
    m_model.layer4.seq_1.bn3.running_mean = t2m_param(t_model.layer4[1].bn3.running_mean)
    m_model.layer4.seq_1.bn3.running_var = t2m_param(t_model.layer4[1].bn3.running_var)
    m_model.layer4.seq_2.conv1.w = t2m_param(t_model.layer4[2].conv1.weight)
    m_model.layer4.seq_2.bn1.w = t2m_param(t_model.layer4[2].bn1.weight)
    m_model.layer4.seq_2.bn1.b = t2m_param(t_model.layer4[2].bn1.bias)
    m_model.layer4.seq_2.bn1.running_mean = t2m_param(t_model.layer4[2].bn1.running_mean)
    m_model.layer4.seq_2.bn1.running_var = t2m_param(t_model.layer4[2].bn1.running_var)
    m_model.layer4.seq_2.conv2.w = t2m_param(t_model.layer4[2].conv2.weight)
    m_model.layer4.seq_2.bn2.w = t2m_param(t_model.layer4[2].bn2.weight)
    m_model.layer4.seq_2.bn2.b = t2m_param(t_model.layer4[2].bn2.bias)
    m_model.layer4.seq_2.bn2.running_mean = t2m_param(t_model.layer4[2].bn2.running_mean)
    m_model.layer4.seq_2.bn2.running_var = t2m_param(t_model.layer4[2].bn2.running_var)
    m_model.layer4.seq_2.conv3.w = t2m_param(t_model.layer4[2].conv3.weight)
    m_model.layer4.seq_2.bn3.w = t2m_param(t_model.layer4[2].bn3.weight)
    m_model.layer4.seq_2.bn3.b = t2m_param(t_model.layer4[2].bn3.bias)
    m_model.layer4.seq_2.bn3.running_mean = t2m_param(t_model.layer4[2].bn3.running_mean)
    m_model.layer4.seq_2.bn3.running_var = t2m_param(t_model.layer4[2].bn3.running_var)

    m_model.fc1.w = t2m_param(t_model.fc1.weight)
    m_model.fc1.b = t2m_param(t_model.fc1.bias)
    m_x, t_x = randn_torch(
        [1, 3, 224, 224], requires_grad=True, device="cuda"
    )  # pylint: disable=unused-variable
    m_y, t_y = one_hot_torch(batch_size=1, num_classes=1000, device="cuda")
    m_x.requires_grad = True
    m_model.train_mode()
    t_model.train()
    m_loss = m_model(m_x, m_y)
    t_loss = t_model(t_x, t_y)
    m_loss.backward()
    t_loss.backward()

    check(m_loss, t_loss)
    check(m_model.bn1.running_mean, t_model.bn1.running_mean, atol=1e-3, rtol=1e-3)
    check(m_model.bn1.running_var, t_model.bn1.running_var, atol=1e-3, rtol=1e-3)
    check(
        m_model.layer1.seq_0.bn1.running_mean,
        t_model.layer1[0].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.bn1.running_var,
        t_model.layer1[0].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.bn2.running_mean,
        t_model.layer1[0].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.bn2.running_var,
        t_model.layer1[0].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.bn3.running_mean,
        t_model.layer1[0].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.bn3.running_var,
        t_model.layer1[0].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.downsample.seq_1.running_mean,
        t_model.layer1[0].downsample[1].running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_0.downsample.seq_1.running_var,
        t_model.layer1[0].downsample[1].running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn1.running_mean,
        t_model.layer1[1].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn1.running_var,
        t_model.layer1[1].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn2.running_mean,
        t_model.layer1[1].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn2.running_var,
        t_model.layer1[1].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn3.running_mean,
        t_model.layer1[1].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_1.bn3.running_var,
        t_model.layer1[1].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn1.running_mean,
        t_model.layer1[2].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn1.running_var,
        t_model.layer1[2].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn2.running_mean,
        t_model.layer1[2].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn2.running_var,
        t_model.layer1[2].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn3.running_mean,
        t_model.layer1[2].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer1.seq_2.bn3.running_var,
        t_model.layer1[2].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn1.running_mean,
        t_model.layer2[0].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn1.running_var,
        t_model.layer2[0].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn2.running_mean,
        t_model.layer2[0].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn2.running_var,
        t_model.layer2[0].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn3.running_mean,
        t_model.layer2[0].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.bn3.running_var,
        t_model.layer2[0].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.downsample.seq_1.running_mean,
        t_model.layer2[0].downsample[1].running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_0.downsample.seq_1.running_var,
        t_model.layer2[0].downsample[1].running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn1.running_mean,
        t_model.layer2[1].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn1.running_var,
        t_model.layer2[1].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn2.running_mean,
        t_model.layer2[1].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn2.running_var,
        t_model.layer2[1].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn3.running_mean,
        t_model.layer2[1].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_1.bn3.running_var,
        t_model.layer2[1].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn1.running_mean,
        t_model.layer2[2].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn1.running_var,
        t_model.layer2[2].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn2.running_mean,
        t_model.layer2[2].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn2.running_var,
        t_model.layer2[2].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn3.running_mean,
        t_model.layer2[2].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_2.bn3.running_var,
        t_model.layer2[2].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn1.running_mean,
        t_model.layer2[3].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn1.running_var,
        t_model.layer2[3].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn2.running_mean,
        t_model.layer2[3].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn2.running_var,
        t_model.layer2[3].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn3.running_mean,
        t_model.layer2[3].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer2.seq_3.bn3.running_var,
        t_model.layer2[3].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn1.running_mean,
        t_model.layer3[0].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn1.running_var,
        t_model.layer3[0].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn2.running_mean,
        t_model.layer3[0].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn2.running_var,
        t_model.layer3[0].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn3.running_mean,
        t_model.layer3[0].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.bn3.running_var,
        t_model.layer3[0].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.downsample.seq_1.running_mean,
        t_model.layer3[0].downsample[1].running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_0.downsample.seq_1.running_var,
        t_model.layer3[0].downsample[1].running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn1.running_mean,
        t_model.layer3[1].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn1.running_var,
        t_model.layer3[1].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn2.running_mean,
        t_model.layer3[1].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn2.running_var,
        t_model.layer3[1].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn3.running_mean,
        t_model.layer3[1].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_1.bn3.running_var,
        t_model.layer3[1].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn1.running_mean,
        t_model.layer3[2].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn1.running_var,
        t_model.layer3[2].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn2.running_mean,
        t_model.layer3[2].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn2.running_var,
        t_model.layer3[2].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn3.running_mean,
        t_model.layer3[2].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_2.bn3.running_var,
        t_model.layer3[2].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn1.running_mean,
        t_model.layer3[3].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn1.running_var,
        t_model.layer3[3].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn2.running_mean,
        t_model.layer3[3].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn2.running_var,
        t_model.layer3[3].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn3.running_mean,
        t_model.layer3[3].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_3.bn3.running_var,
        t_model.layer3[3].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn1.running_mean,
        t_model.layer3[4].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn1.running_var,
        t_model.layer3[4].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn2.running_mean,
        t_model.layer3[4].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn2.running_var,
        t_model.layer3[4].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn3.running_mean,
        t_model.layer3[4].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_4.bn3.running_var,
        t_model.layer3[4].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn1.running_mean,
        t_model.layer3[5].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn1.running_var,
        t_model.layer3[5].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn2.running_mean,
        t_model.layer3[5].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn2.running_var,
        t_model.layer3[5].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn3.running_mean,
        t_model.layer3[5].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer3.seq_5.bn3.running_var,
        t_model.layer3[5].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn1.running_mean,
        t_model.layer4[0].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn1.running_var,
        t_model.layer4[0].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn2.running_mean,
        t_model.layer4[0].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn2.running_var,
        t_model.layer4[0].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn3.running_mean,
        t_model.layer4[0].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.bn3.running_var,
        t_model.layer4[0].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.downsample.seq_1.running_mean,
        t_model.layer4[0].downsample[1].running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_0.downsample.seq_1.running_var,
        t_model.layer4[0].downsample[1].running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn1.running_mean,
        t_model.layer4[1].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn1.running_var,
        t_model.layer4[1].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn2.running_mean,
        t_model.layer4[1].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn2.running_var,
        t_model.layer4[1].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn3.running_mean,
        t_model.layer4[1].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_1.bn3.running_var,
        t_model.layer4[1].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn1.running_mean,
        t_model.layer4[2].bn1.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn1.running_var,
        t_model.layer4[2].bn1.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn2.running_mean,
        t_model.layer4[2].bn2.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn2.running_var,
        t_model.layer4[2].bn2.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn3.running_mean,
        t_model.layer4[2].bn3.running_mean,
        atol=1e-3,
        rtol=1e-3,
    )
    check(
        m_model.layer4.seq_2.bn3.running_var,
        t_model.layer4[2].bn3.running_var,
        atol=1e-3,
        rtol=1e-3,
    )
    # pylint: enable=no-member,line-too-long


if __name__ == "__main__":
    test_r50_v1_imagenet()
