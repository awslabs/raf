import math

import numpy as np

from mnm._core.ndarray import Parameter
from mnm._op.sym import (batch_norm_infer, batch_norm_train, bias_add, conv2d,
                         matmul)
from mnm.random import uniform
from mnm.random.nn import kaiming_uniform

from .model import Model
from .model import script_model as script
from .model import script_mutate_attr as script_mutate


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
        self.w_shape = (in_channels, out_channels // groups, *kernel_size)
        self.b_shape = (out_channels, ) if bias else None
        self.reset()

    def reset(self):
        self.w = Parameter(kaiming_uniform(self.w_shape, name="w"))
        self.b = None
        if self.b_shape is not None:
            _, fan_in, _, _ = self.w_shape
            bound = 1.0 / math.sqrt(fan_in)
            self.b = Parameter(uniform(-bound, bound, self.b_shape, name="b"))

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        x = conv2d(x,
                   self.w,
                   stride=self.stride,
                   padding=self.padding,
                   dilation=self.dilation,
                   groups=self.groups)
        if self.b is not None:
            x = bias_add(x, self.b, axis=1)
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
        self.running_mean = Parameter(np.zeros(n_f, dtype="float32"),
                                      name="running_mean")
        self.running_var = Parameter(np.ones(n_f, dtype="float32"),
                                     name="running_var")
        self.w = None
        self.b = None
        if self.affine:
            self.w = Parameter(np.zeros(n_f, dtype="float32"), name="w")
            self.b = Parameter(np.ones(n_f, dtype="float32"), name="b")

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        ret = batch_norm_train(x=x,
                               w=self.w,
                               b=self.b,
                               running_mean=self.running_mean,
                               running_var=self.running_var,
                               eps=self.eps,
                               momentum=self.momentum)
        script_mutate(self, "running_mean", ret[1])
        script_mutate(self, "running_var", ret[2])
        return ret[0]

    @script
    def forward_infer(self, x):
        ret = batch_norm_infer(x=x,
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
        self.w = Parameter(
            kaiming_uniform((self.out_features, self.in_features), name="w"))
        self.b = None
        if self.bias:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            self.b = Parameter(uniform(-bound, bound, [self.out_features]),
                               name="b")

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        out = matmul(x, self.w, transpose_b=True)
        if self.b is not None:
            out = bias_add(out, self.b, axis=-1)
        return out
