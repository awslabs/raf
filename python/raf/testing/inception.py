# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inception V3 model and its blocks.
This model can be used to test the inter-operator parallel execution.
"""
# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals, invalid-name,
# pylint: disable=too-many-instance-attributes, arguments-differ, abstract-method
# pylint: disable=missing-class-docstring, too-many-arguments, missing-function-docstring
from collections import OrderedDict
import itertools

import torch
from torch import nn
from torch.functional import F
import scipy.stats as stats

import raf
import raf.model.nn as mnn
from .common import check, randn_torch, t2m_param, one_hot_torch
from .utils import get_param, set_param


class TorchInception3(nn.Module):
    """adopted from torchvision.models.inception"""

    def __init__(self, num_classes=1000) -> None:
        super(TorchInception3, self).__init__()

        self.Conv2d_1a_3x3 = TorchBasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = TorchBasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = TorchBasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = TorchBasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = TorchBasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = TorchInceptionA(192, pool_features=32)
        self.Mixed_5c = TorchInceptionA(256, pool_features=64)
        self.Mixed_5d = TorchInceptionA(288, pool_features=64)
        self.Mixed_6a = TorchInceptionB(288)
        self.Mixed_6b = TorchInceptionC(768, channels_7x7=128)
        self.Mixed_6c = TorchInceptionC(768, channels_7x7=160)
        self.Mixed_6d = TorchInceptionC(768, channels_7x7=160)
        self.Mixed_6e = TorchInceptionC(768, channels_7x7=192)
        self.Mixed_7a = TorchInceptionD(768)
        self.Mixed_7b = TorchInceptionE(1280)
        self.Mixed_7c = TorchInceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                stddev = m.stddev if hasattr(m, "stddev") else 0.1
                x = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(x.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y_true):  # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = self.maxpool1(x)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = self.maxpool2(x)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        x = self.avgpool(x)  # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)  # N x 2048
        x = self.fc(x)  # N x 1000 (num_classes)
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


class TorchInceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(TorchInceptionA, self).__init__()
        self.branch1x1 = TorchBasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = TorchBasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = TorchBasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = TorchBasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = TorchBasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = TorchBasicConv2d(96, 96, kernel_size=3, padding=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = TorchBasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class TorchInceptionB(nn.Module):
    def __init__(self, in_channels):
        super(TorchInceptionB, self).__init__()
        self.branch3x3 = TorchBasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = TorchBasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = TorchBasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = TorchBasicConv2d(96, 96, kernel_size=3, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.max_pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class TorchInceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(TorchInceptionC, self).__init__()
        self.branch1x1 = TorchBasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = TorchBasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = TorchBasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = TorchBasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = TorchBasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = TorchBasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = TorchBasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = TorchBasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = TorchBasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = TorchBasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class TorchInceptionD(nn.Module):
    def __init__(self, in_channels):
        super(TorchInceptionD, self).__init__()
        self.branch3x3_1 = TorchBasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = TorchBasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = TorchBasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = TorchBasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = TorchBasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = TorchBasicConv2d(192, 192, kernel_size=3, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.max_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class TorchInceptionE(nn.Module):
    def __init__(self, in_channels):
        super(TorchInceptionE, self).__init__()
        self.branch1x1 = TorchBasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = TorchBasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = TorchBasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = TorchBasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = TorchBasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = TorchBasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = TorchBasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = TorchBasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = TorchBasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class TorchInceptionAB(nn.Module):
    def __init__(self):
        super(TorchInceptionAB, self).__init__()
        self.block_a = TorchInceptionA(192, pool_features=64)
        self.block_b = TorchInceptionB(288)

    def forward(self, x):
        x = self.block_a(x)
        return self.block_b(x)


class TorchInceptionDE(nn.Module):
    def __init__(self):
        super(TorchInceptionDE, self).__init__()
        self.block_d = TorchInceptionD(768)
        self.block_e = TorchInceptionE(1280)

    def forward(self, x):
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class TorchInceptionCD(nn.Module):
    def __init__(self):
        super(TorchInceptionCD, self).__init__()
        self.block_c = TorchInceptionC(768, 192)
        self.block_d = TorchInceptionD(768)

    def forward(self, x):
        x = self.block_c(x)
        x = self.block_d(x)
        return x


class TorchInceptionCDE(nn.Module):
    def __init__(self):
        super(TorchInceptionCDE, self).__init__()
        self.block_c = TorchInceptionC(768, 192)
        self.block_d = TorchInceptionD(768)
        self.block_e = TorchInceptionE(1280)

    def forward(self, x):
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class TorchInceptionABCDE(nn.Module):
    def __init__(self):
        super(TorchInceptionABCDE, self).__init__()
        self.block_a = TorchInceptionA(192, pool_features=64)
        self.block_b = TorchInceptionB(288)
        self.block_c = TorchInceptionC(768, 192)
        self.block_d = TorchInceptionD(768)
        self.block_e = TorchInceptionE(1280)

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class TorchBasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(TorchBasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TorchInceptionBlock(nn.Module):
    def __init__(self, block_class, out_channels, *args, **kwargs):
        super(TorchInceptionBlock, self).__init__()
        self.block = block_class(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, 1000)

    def forward(self, x, y_true):
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


class RAFInception3(raf.Model):
    """adopted from torchvision.models.inception"""

    def build(self, num_classes=1000) -> None:
        self.Conv2d_1a_3x3 = RAFBasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = RAFBasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = RAFBasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = RAFMaxPool2d(kernel=3, stride=2)
        self.Conv2d_3b_1x1 = RAFBasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = RAFBasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = RAFMaxPool2d(kernel=3, stride=2)
        self.Mixed_5b = RAFInceptionA(192, pool_features=32)
        self.Mixed_5c = RAFInceptionA(256, pool_features=64)
        self.Mixed_5d = RAFInceptionA(288, pool_features=64)
        self.Mixed_6a = RAFInceptionB(288)
        self.Mixed_6b = RAFInceptionC(768, channels_7x7=128)
        self.Mixed_6c = RAFInceptionC(768, channels_7x7=160)
        self.Mixed_6d = RAFInceptionC(768, channels_7x7=160)
        self.Mixed_6e = RAFInceptionC(768, channels_7x7=192)
        self.Mixed_7a = RAFInceptionD(768)
        self.Mixed_7b = RAFInceptionE(1280)
        self.Mixed_7c = RAFInceptionE(2048)
        self.avgpool = RAFAdaptiveAvgPool2d((1, 1))
        self.fc = mnn.Linear(2048, num_classes)

    @raf.model.trace
    def forward_infer(self, x):  # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)  # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)  # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)  # N x 64 x 147 x 147
        x = self.maxpool1(x)  # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)  # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)  # N x 192 x 71 x 71
        x = self.maxpool2(x)  # N x 192 x 35 x 35
        x = self.Mixed_5b(x)  # N x 256 x 35 x 35
        x = self.Mixed_5c(x)  # N x 288 x 35 x 35
        x = self.Mixed_5d(x)  # N x 288 x 35 x 35
        x = self.Mixed_6a(x)  # N x 768 x 17 x 17
        x = self.Mixed_6b(x)  # N x 768 x 17 x 17
        x = self.Mixed_6c(x)  # N x 768 x 17 x 17
        x = self.Mixed_6d(x)  # N x 768 x 17 x 17
        x = self.Mixed_6e(x)  # N x 768 x 17 x 17
        x = self.Mixed_7a(x)  # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)  # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)  # N x 2048 x 8 x 8
        x = self.avgpool(x)  # N x 2048 x 1 x 1
        x = raf.batch_flatten(x)
        # x = torch.flatten(x, 1)  # N x 2048
        x = self.fc(x)  # N x 1000 (num_classes)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        x = self.forward_infer(x)
        y_pred = raf.log_softmax(x)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss


class RAFInceptionA(raf.Model):
    def build(self, in_channels, pool_features):
        self.branch1x1 = RAFBasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = RAFBasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = RAFBasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = RAFBasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = RAFBasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = RAFBasicConv2d(96, 96, kernel_size=3, padding=1)

        self.avg_pool = RAFAvgPool2d(kernel=3, stride=1, padding=1)
        self.branch_pool = RAFBasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return raf.concatenate(outputs, 1)


class RAFInceptionB(raf.Model):
    def build(self, in_channels):
        self.branch3x3 = RAFBasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = RAFBasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = RAFBasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = RAFBasicConv2d(96, 96, kernel_size=3, stride=2)
        self.max_pool = RAFMaxPool2d(kernel=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.max_pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return raf.concatenate(outputs, 1)


class RAFInceptionC(raf.Model):
    def build(self, in_channels, channels_7x7):
        self.branch1x1 = RAFBasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = RAFBasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = RAFBasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = RAFBasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = RAFBasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = RAFBasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = RAFBasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = RAFBasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = RAFBasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.avg_pool = RAFAvgPool2d(kernel=3, stride=1, padding=1)
        self.branch_pool = RAFBasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return raf.concatenate(outputs, 1)
        # return torch.cat(outputs, 1)


class RAFInceptionD(raf.Model):
    def build(self, in_channels):
        self.branch3x3_1 = RAFBasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = RAFBasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = RAFBasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = RAFBasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = RAFBasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = RAFBasicConv2d(192, 192, kernel_size=3, stride=2)
        self.max_pool = RAFMaxPool2d(kernel=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.max_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return raf.concatenate(outputs, 1)
        # return torch.cat(outputs, 1)


class RAFInceptionE(raf.Model):
    def build(self, in_channels):
        self.branch1x1 = RAFBasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = RAFBasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = RAFBasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = RAFBasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = RAFBasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = RAFBasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = RAFBasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = RAFBasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool = RAFAvgPool2d(kernel=3, stride=1, padding=1)
        self.branch_pool = RAFBasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        # branch3x3 = torch.cat(branch3x3, 1)
        branch3x3 = raf.concatenate(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        # branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch3x3dbl = raf.concatenate(branch3x3dbl, 1)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # return torch.cat(outputs, 1)
        return raf.concatenate(outputs, 1)


class RAFInceptionAB(raf.Model):
    def build(self):
        self.block_a = RAFInceptionA(192, pool_features=64)
        self.block_b = RAFInceptionB(288)

    def forward(self, x):
        x = self.block_a(x)
        return self.block_b(x)


class RAFInceptionCD(raf.Model):
    def build(self):
        self.block_c = RAFInceptionC(768, 192)
        self.block_d = RAFInceptionD(768)

    def forward(self, x):
        x = self.block_c(x)
        x = self.block_d(x)
        return x


class RAFInceptionCDE(raf.Model):
    def build(self):
        self.block_c = RAFInceptionC(768, 192)
        self.block_d = RAFInceptionD(768)
        self.block_e = RAFInceptionE(1280)

    def forward(self, x):
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class RAFInceptionDE(raf.Model):
    def build(self):
        self.block_d = RAFInceptionD(768)
        self.block_e = RAFInceptionE(1280)

    def forward(self, x):
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class RAFInceptionABCDE(raf.Model):
    def build(self):
        self.block_a = RAFInceptionA(192, pool_features=64)
        self.block_b = RAFInceptionB(288)
        self.block_c = RAFInceptionC(768, 192)
        self.block_d = RAFInceptionD(768)
        self.block_e = RAFInceptionE(1280)

    def forward(self, x):
        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        return x


class RAFBasicConv2d(raf.Model):
    def build(self, in_channels, out_channels, **kwargs):
        self.conv = mnn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = mnn.BatchNorm(out_channels, eps=0.001)
        self.relu = RAFReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class RAFInceptionBlock(raf.Model):
    def build(self, block_class, out_channels, *args, **kwargs):
        self.block = block_class(*args, **kwargs)
        self.avgpool = RAFAdaptiveAvgPool2d((1, 1))
        self.fc = mnn.Linear(out_channels, 1000)

    @raf.model.trace
    def forward_infer(self, x):
        x = self.block(x)
        x = self.avgpool(x)
        x = raf.batch_flatten(x)
        x = self.fc(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss


class RAFMaxPool2d(raf.Model):
    def build(
        self,
        kernel,
        stride,
        padding=0,
        dilation=1,
        ceil_mode=False,
        include_pad=True,
        layout="NCHW",
    ):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.include_pad = include_pad
        self.layout = layout

    @raf.model.trace
    def forward(self, x):
        return raf.max_pool2d(
            x,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.include_pad,
            self.layout,
        )


class RAFAvgPool2d(raf.Model):
    def build(
        self,
        kernel,
        stride,
        padding=0,
        dilation=1,
        ceil_mode=False,
        include_pad=True,
        layout="NCHW",
    ):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.include_pad = include_pad
        self.layout = layout

    @raf.model.trace
    def forward(self, x):
        return raf.avg_pool2d(
            x,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.include_pad,
            self.layout,
        )


class RAFAdaptiveAvgPool2d(raf.Model):
    def build(self, shape, layout="NCHW"):
        self.shape = shape
        self.layout = layout

    @raf.model.trace
    def forward(self, x):
        return raf.adaptive_avg_pool2d(x, self.shape, self.layout)


class RAFReLU(raf.Model):
    def build(self):
        pass

    @raf.model.trace
    def forward(self, x):
        # pylint: disable=no-self-use
        return raf.relu(x)


def torch_named_params(t_module: nn.Module):
    """return an OrderDict that maps the parameter and buffer name to torch tensor"""
    named_params = list(t_module.named_parameters())
    named_buffers = list(t_module.named_buffers())
    return OrderedDict(itertools.chain(named_params, named_buffers))
    # return OrderedDict(itertools.chain(t_module.named_parameters(), t_module.named_buffers()))


def raf_named_params(m_model: raf.Model):
    """return an OrderDict that maps the parameter name to raf tensor"""
    return m_model.state()


def param_map(m_model, t_model):
    """maps from m_model parameter name to t_model parameter value"""
    t_param_dict = torch_named_params(t_model)
    m_param_dict = raf_named_params(m_model)
    result = OrderedDict()
    tail_rename_map = {".b": ".bias", ".w": ".weight"}
    for m_name in m_param_dict:
        assert isinstance(m_name, str)
        m_name_tail = m_name[m_name.rfind(".") :]
        if m_name_tail in tail_rename_map:
            t_name = m_name[: m_name.rfind(".")] + tail_rename_map[m_name_tail]
        else:
            t_name = m_name
        assert t_name in t_param_dict
        result[m_name] = t_param_dict[t_name]
        del t_param_dict[t_name]
    # make sure the remaining parameters in torch module is num_batches_tracked in nn.BatchNorm2d.
    for t_name in t_param_dict:
        assert t_name.endswith(".num_batches_tracked")
    return result


def check_params(m_model, t_model, atol=1e-3, rtol=1e-3):
    """check the parameters of m_model and t_model"""
    for m_name, t_w in param_map(m_model, t_model).items():
        m_w = get_param(m_model, m_name)
        try:
            check(m_w, t_w, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"Weight {m_name} check failed.")
            raise e


def init(m_model, t_model, device="cpu"):
    """initialize raf model with parameters of torch model"""
    for m_name, t_w in param_map(m_model, t_model).items():
        set_param(m_model, m_name, t2m_param(t_w, device=device))


def get_input(batch_size=1, device="cuda"):
    """get inception input"""
    m_x, t_x = randn_torch([batch_size, 3, 299, 299], device=device, requires_grad=True)
    m_y, t_y = one_hot_torch(batch_size, num_classes=1000, device=device)
    return [(m_x, m_y), (t_x, t_y)]


def get_model():
    """get inception v3 model"""
    m_model = RAFInception3()
    t_model = TorchInception3()
    init(m_model, t_model)
    m_model.train_mode()
    t_model.train()
    return m_model, t_model


def get_block_and_input(block_name: str, device="cpu", batch_size=1):
    """
    Get the inception block and its input.
    The input equals to the input of its first occurrence in inception v3.
    """
    block_name = block_name.upper()
    m_block_args = {
        "A": {
            "block_class": RAFInceptionA,
            "in_channels": 192,
            "out_channels": 256,
            "pool_features": 32,
        },
        "B": {"block_class": RAFInceptionB, "in_channels": 288, "out_channels": 768},
        "C": {
            "block_class": RAFInceptionC,
            "in_channels": 768,
            "out_channels": 768,
            "channels_7x7": 128,
        },
        "D": {"block_class": RAFInceptionD, "in_channels": 768, "out_channels": 1280},
        "E": {"block_class": RAFInceptionE, "in_channels": 1280, "out_channels": 2048},
        "AB": {"block_class": RAFInceptionAB, "out_channels": 768},
        "CD": {"block_class": RAFInceptionCD, "out_channels": 1280},
        "DE": {"block_class": RAFInceptionDE, "out_channels": 2048},
        "CDE": {"block_class": RAFInceptionCDE, "out_channels": 2048},
        "ABCDE": {"block_class": RAFInceptionABCDE, "out_channels": 2048},
    }
    t_block_args = {
        "A": {
            "block_class": TorchInceptionA,
            "in_channels": 192,
            "out_channels": 256,
            "pool_features": 32,
        },
        "B": {"block_class": TorchInceptionB, "in_channels": 288, "out_channels": 768},
        "C": {
            "block_class": TorchInceptionC,
            "in_channels": 768,
            "out_channels": 768,
            "channels_7x7": 128,
        },
        "D": {"block_class": TorchInceptionD, "in_channels": 768, "out_channels": 1280},
        "E": {"block_class": TorchInceptionE, "in_channels": 1280, "out_channels": 2048},
        "AB": {"block_class": TorchInceptionAB, "out_channels": 768},
        "CD": {"block_class": TorchInceptionCD, "out_channels": 1280},
        "DE": {"block_class": TorchInceptionDE, "out_channels": 2048},
        "CDE": {"block_class": TorchInceptionCDE, "out_channels": 2048},
        "ABCDE": {"block_class": TorchInceptionABCDE, "out_channels": 2048},
    }
    input_shapes = {
        "A": (batch_size, 192, 35, 35),
        "B": (batch_size, 288, 35, 35),
        "C": (batch_size, 768, 17, 17),
        "D": (batch_size, 768, 17, 17),
        "E": (batch_size, 1280, 8, 8),
        "AB": (batch_size, 192, 35, 35),
        "DE": (batch_size, 768, 17, 17),
        "CD": (batch_size, 768, 17, 17),
        "CDE": (batch_size, 768, 17, 17),
        "ABCDE": (batch_size, 192, 35, 35),
    }
    m_block = RAFInceptionBlock(**m_block_args[block_name])
    t_block = TorchInceptionBlock(**t_block_args[block_name])
    init(m_block, t_block)
    m_block.train_mode()
    m_block.to(device=device)
    t_block.train()
    t_block.to(device=device)
    m_x, t_x = randn_torch(input_shapes[block_name], device=device, requires_grad=True)
    m_y, t_y = one_hot_torch(batch_size, num_classes=1000, device=device)
    return [(m_block, m_x, m_y), (t_block, t_x, t_y)]
