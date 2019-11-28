import numpy as np

from mnm._core.model import Model
from mnm._core.ndarray import Parameter
from mnm._core.script import script_model as script
from mnm._core.script import script_mutate_attr as script_mutate


class Conv2d(Model):

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
        w = np.zeros((out_channels, in_channels // groups, kernel_size[0],
                      kernel_size[1]),
                     dtype="float32")
        self.w = Parameter(w, name="w")
        if bias:
            b = np.zeros((1, out_channels, 1, 1), dtype="float32")
            self.b = Parameter(b, name="b")
        else:
            self.b = None

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        import mnm  # pylint: disable=import-outside-toplevel
        x = mnm.conv2d(x,
                       self.w,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups)
        if self.b is not None:
            x = mnm.add(x, self.b)
        return x


class BatchNorm(Model):

    # pylint: disable=attribute-defined-outside-init
    def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = Parameter(np.zeros(num_features, dtype="float32"),
                                      name="running_mean")
        self.running_var = Parameter(np.ones(num_features, dtype="float32"),
                                     name="running_var")
        if affine:
            self.w = Parameter(shape=(num_features, ), name="w")
            self.b = Parameter(shape=(num_features, ), name="b")
        else:
            self.w = None
            self.b = None

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        import mnm  # pylint: disable=import-outside-toplevel
        ret = mnm.batch_norm_train(x=x,
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
        import mnm  # pylint: disable=import-outside-toplevel
        ret = mnm.batch_norm_infer(x=x,
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
        w = np.zeros((out_features, in_features), dtype="float32")
        self.w = Parameter(w, name="w")
        if bias:
            b = np.zeros((out_features, ), dtype="float32")
            self.b = Parameter(b, name="b")
        else:
            self.b = None

    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        import mnm  # pylint: disable=import-outside-toplevel
        out = mnm.linear(x, self.w)
        if self.b is not None:
            out = mnm.add(out, self.b)
        return out
