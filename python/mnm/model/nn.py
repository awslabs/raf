import math

import numpy as np

from mnm._core.ndarray import ndarray
from mnm._op import sym
from mnm.random import uniform
from mnm.random.nn import kaiming_uniform

from .model import Model
from .trace import trace, trace_mutate_attr


class Conv2d(Model):  # pylint: disable=too-many-instance-attributes

    # pylint: disable=attribute-defined-outside-init
    def build(  # pylint: disable=too-many-arguments
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.w_shape = (out_channels, in_channels // groups, *kernel_size)
        self.b_shape = (out_channels, 1, 1) if bias else None
        self.reset()

    def reset(self):
        self.w = ndarray(kaiming_uniform(self.w_shape, name="w"))
        self.b = None
        if self.b_shape is not None:
            _, fan_in, _, _ = self.w_shape
            bound = 1.0 / math.sqrt(fan_in)
            self.b = ndarray(uniform(-bound, bound, self.b_shape, name="b"))

    # pylint: enable=attribute-defined-outside-init

    @trace
    def forward(self, x):
        x = sym.conv2d(x,
                       self.w,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups)
        if self.b is not None:
            x = sym.add(x, self.b)
        return x


class BatchNorm(Model):  # pylint: disable=too-many-instance-attributes

    # pylint: disable=attribute-defined-outside-init
    def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.reset()

    def reset(self):
        n_f = self.num_features
        self.running_mean = ndarray(np.zeros(n_f, dtype="float32"),
                                    name="running_mean")
        self.running_var = ndarray(np.ones(n_f, dtype="float32"),
                                   name="running_var")
        self.w = None
        self.b = None
        if self.affine:
            self.w = ndarray(np.zeros(n_f, dtype="float32"), name="w")
            self.b = ndarray(np.ones(n_f, dtype="float32"), name="b")

    # pylint: enable=attribute-defined-outside-init

    @trace
    def forward(self, x):
        ret = sym.batch_norm_train(x=x,
                                   w=self.w,
                                   b=self.b,
                                   running_mean=self.running_mean,
                                   running_var=self.running_var,
                                   eps=self.eps,
                                   momentum=self.momentum)
        trace_mutate_attr(self, "running_mean", ret[1])
        trace_mutate_attr(self, "running_var", ret[2])
        return ret[0]

    @trace
    def forward_infer(self, x):
        ret = sym.batch_norm_infer(x=x,
                                   w=self.w,
                                   b=self.b,
                                   running_mean=self.running_mean,
                                   running_var=self.running_var,
                                   eps=self.eps,
                                   momentum=self.momentum)
        return ret


class Linear(Model):

    # pylint: disable=attribute-defined-outside-init
    def build(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.reset()

    def reset(self):
        self.w = ndarray(
            kaiming_uniform((self.out_features, self.in_features), name="w"))
        self.b = None
        if self.bias:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            self.b = ndarray(uniform(-bound, bound, [self.out_features]),
                             name="b")

    # pylint: enable=attribute-defined-outside-init

    @trace
    def forward(self, x):
        out = sym.matmul_nt(x, self.w)
        if self.b is not None:
            out = sym.add(out, self.b)
        return out
