# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring,attribute-defined-outside-init
"""Neural network specific Model blocks."""
import math

import numpy as np

from raf._core.ndarray import ndarray, array
from raf._core.core_utils import get_chained_attr
from raf._op import sym
from raf.random import uniform
from raf.random.nn import kaiming_uniform

from .model import Model
from .trace import trace, trace_mutate_attr  # pylint: disable=unused-import


class Conv2d(Model):  # pylint: disable=too-many-instance-attributes
    def build(  # pylint: disable=too-many-arguments
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        channel_mode="NCHW",
    ):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if channel_mode == "NCHW":
            self.w_shape = (out_channels, in_channels // groups, *kernel_size)
            self.channel_configs = {"layout": "NCHW", "kernel_layout": "OIHW", "out_layout": "NCHW"}
        elif channel_mode == "NHWC":
            self.channel_configs = {"layout": "NHWC", "kernel_layout": "OHWI", "out_layout": "NHWC"}
            self.w_shape = (out_channels, *kernel_size, in_channels // groups)
        else:
            raise ValueError("Unknown channel mode: " + channel_mode)
        self.b_shape = (out_channels,) if bias else None
        self.b = None
        self.reset()

    def reset(self):
        self.w = kaiming_uniform(self.w_shape, name="w")
        if self.b_shape is not None:
            _, fan_in, _, _ = self.w_shape
            bound = 1.0 / math.sqrt(fan_in)
            self.b = uniform(
                -bound,
                bound,
                self.b_shape,
                name="b",
                device=get_chained_attr(self, ["b", "device"], default="cpu"),
            )

    @trace
    def forward(self, x):
        x = sym.conv2d(
            x,
            self.w,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            **self.channel_configs
        )
        if self.b is not None:
            x = sym.bias_add(x, self.b)
        return x


class BatchNorm(Model):  # pylint: disable=too-many-instance-attributes
    def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_var = self.running_mean = None
        if affine:
            self.w = self.b = None
        self.reset()

    def reset(self):
        n_f = self.num_features
        self.running_mean = ndarray(
            np.zeros(n_f, dtype="float32"),
            name="running_mean",
            device=get_chained_attr(self, ["running_mean", "device"], "cpu"),
        )
        self.running_var = ndarray(
            np.ones(n_f, dtype="float32"),
            name="running_var",
            device=get_chained_attr(self, ["running_var", "device"], "cpu"),
        )
        if self.affine:
            self.w = ndarray(
                np.ones(n_f, dtype="float32"),
                name="w",
                device=get_chained_attr(self, ["w", "device"], "cpu"),
            )
            self.w.requires_grad = True
            self.b = ndarray(
                np.zeros(n_f, dtype="float32"),
                name="b",
                device=get_chained_attr(self, ["b", "device"], "cpu"),
            )
            self.b.requires_grad = True

    @trace
    def forward(self, x):
        ret = sym.batch_norm_train(
            x=x,
            w=self.w,
            b=self.b,
            running_mean=self.running_mean,
            running_var=self.running_var,
            eps=self.eps,
            momentum=self.momentum,
        )
        trace_mutate_attr(self, "running_mean", ret[1])
        trace_mutate_attr(self, "running_var", ret[2])
        return ret[0]

    @trace
    def forward_infer(self, x):
        ret = sym.batch_norm_infer(
            x=x,
            w=self.w,
            b=self.b,
            running_mean=self.running_mean,
            running_var=self.running_var,
            eps=self.eps,
            momentum=self.momentum,
        )
        return ret


class Linear(Model):
    def build(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.b = None
        self.reset()

    def reset(self):
        self.w = kaiming_uniform(
            (self.out_features, self.in_features),
            name="w",
            device=get_chained_attr(self, ["w", "device"], "cpu"),
        )
        if self.bias:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            self.b = uniform(
                -bound,
                bound,
                [self.out_features],
                name="b",
                device=get_chained_attr(self, ["b", "device"], "cpu"),
            )

    @trace
    def forward(self, x):
        out = sym.dense(x, self.w)
        if self.bias:
            out = sym.add(out, self.b)
        return out


class GELU(Model):
    def build(self):
        self.reset()

    def reset(self):
        self._inv_sqrt_2 = array(
            1 / math.sqrt(2),
            name="inv_sqrt_2",
            dtype="float32",
            device=get_chained_attr(self, ["_inv_sqrt_2", "device"], "cpu"),
        )
        self._inv_2 = array(
            1 / 2,
            name="_inv_2",
            dtype="float32",
            device=get_chained_attr(self, ["_inv_2", "device"], "cpu"),
        )

    @trace
    def forward(self, x):
        return sym.multiply(
            x,
            sym.add(
                self._inv_2,
                sym.multiply(sym.erf(sym.multiply(x, self._inv_sqrt_2)), self._inv_2),
            ),
        )
