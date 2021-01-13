import pytest
import numpy as np
import torch
import torch.nn as nn

import mnm
from mnm.model import BatchNorm, Conv2d, Linear, Sequential
from mnm.testing import run_infer_type, run_vm_model, check, randn_torch, get_ctx_list, asnumpy
from mnm._core.core_utils import get_chained_attr


def param_map(t_model, num_blocks):
    names = ["conv1.w", "linear.w", "linear.b"]
    values = [t_model.conv1.weight, t_model.linear.weight, t_model.linear.bias]
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
        "shortcut.w"
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
        "shortcut.weight"
    ]
    for i, num in enumerate(num_blocks):
        for j in range(num):
            m_names, t_names = m_bottleneck_names, t_bottleneck_names
            if j > 0:
                m_names, t_names = m_names[:-1], t_names[:-1]
            prefix = f'layer{i + 1}.seq_{j}'
            cur = [prefix + '.' + name for name in m_names]
            names.extend(cur)
            layer = getattr(t_model, f'layer{i + 1}')
            layer = layer[j]
            cur = [get_param(layer, name) for name in t_names]
            values.extend(cur)
    return dict(zip(names, values))


def get_param(model, name):
    if isinstance(name, str):
        name = name.split('.')
    ret = get_chained_attr(model, name)
    if ret is None:
        raise AttributeError(f"No attribute {name}")
    return ret


def set_param(model, name, value):
    if isinstance(name, str):
        name = name.split('.')
    assert len(name) > 0
    ins = get_param(model, name[:-1])
    setattr(ins, name[-1], value)


def init(m_model, t_model, layers, ctx="cuda"):
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in param_map(t_model, layers).items():
        set_param(m_model, m_name, mnm.array(asnumpy(t_w), ctx=ctx))


def check_params(m_model, t_model, layers):
    # pylint: disable=no-member, line-too-long, too-many-statements
    for m_name, t_w in param_map(t_model, layers).items():
        m_w = get_param(m_model, m_name)
        check(m_w, t_w, atol=1e-3, rtol=1e-3)


def one_hot(batch_size, num_classes, ctx="cuda", dtype="float32"):
    # pylint: disable=not-callable
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    t_x = torch.tensor(targets, requires_grad=False, device=ctx)
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x


class MNMBottleNeck(mnm.Model):
    expansion = 4
    # pylint: disable=attribute-defined-outside-init
    def build(self, in_planes, planes, stride=1):
        self.bn1 = BatchNorm(in_planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv2 = Conv2d(planes,
                            planes,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn3 = BatchNorm(planes)
        self.conv3 = Conv2d(planes,
                            self.expansion * planes,
                            kernel_size=1,
                            bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Conv2d(in_planes,
                                   self.expansion * planes,
                                   kernel_size=1,
                                   stride=stride,
                                   bias=False)
        else:
            self.shortcut = None

    @mnm.model.trace
    def forward(self, x):
        out = mnm.relu(self.bn1(x))

        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(mnm.relu(self.bn2(out)))
        out = self.conv3(mnm.relu(self.bn3(out)))
        out = mnm.add(out, shortcut)
        return out


class MNMResNet50(mnm.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, num_blocks, num_classes=10):
        self.in_planes = 64
        self.conv1 = Conv2d(3,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = Linear(512 * MNMBottleNeck.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for one_stride in strides:
            layers.append(MNMBottleNeck(self.in_planes, planes, one_stride))
            self.in_planes = planes * MNMBottleNeck.expansion
        return Sequential(*layers)

    @mnm.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = mnm.log_softmax(y_pred)
        loss = mnm.nll_loss(y_true, y_pred)
        return loss

    @mnm.model.trace
    def forward_infer(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = mnm.avg_pool2d(out, 4, 4)
        out = mnm.batch_flatten(out)
        out = self.linear(out)
        return out


class TorchBottleneck(nn.Module):  # pylint: disable=abstract-method
    expansion = 4

    def __init__(self, inplanes, planes, stride):
        super(TorchBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding=1,
                               groups=1,
                               dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes * TorchBottleneck.expansion:
            self.shortcut = nn.Conv2d(inplanes,
                                      planes * TorchBottleneck.expansion,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False)
        else:
            self.shortcut = None

    def forward(self, x):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel
        out = F.relu(self.bn1(x))
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)
        out += shortcut
        return out


class TorchResNet50(nn.Module):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method
    def __init__(self, num_blocks, num_classes=10):
        super(TorchResNet50, self).__init__()
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * TorchBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for one_stride in strides:
            layers.append(TorchBottleneck(self.inplanes, planes, one_stride))
            self.inplanes = planes * TorchBottleneck.expansion
        return nn.Sequential(*layers)


    def forward(self, x, y_true):  # pylint: disable=arguments-differ
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4, 4)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.linear(x)
        y_pred = x
        y_pred = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(y_pred, y_true)
        return loss


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_build():
    x = np.random.randn(1, 3, 32, 32)
    y = np.random.randn(1, 10)
    m_x = mnm.array(x, dtype="float32", ctx="cuda")
    m_y = mnm.array(y, dtype='float32', ctx='cuda')
    model = MNMResNet50([3, 4, 6, 3])
    model.to(ctx='cuda')
    print("### Switch to training mode")
    model.train_mode()
    model(m_x, m_y)
    print("### Switch to infer mode")
    model.infer_mode()
    model(m_x)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_build_fp16():
    x = np.random.randn(1, 3, 32, 32)
    m_x = mnm.array(x, dtype="float32", ctx="cuda")
    model = MNMResNet50([3, 4, 6, 3])
    model.to(ctx='cuda')
    model.infer_mode()
    m_y1 = model(m_x)
    print("### Switch to AMP model")
    amp_model = mnm.amp.autocast(model)
    m_y2 = amp_model(m_x)
    np.testing.assert_allclose(m_y1.asnumpy(), m_y2.asnumpy(), rtol=0.1, atol=0.1)


@pytest.mark.parametrize("ctx", get_ctx_list())
@pytest.mark.parametrize("fuse", [False, True])
def test_vm_forward(ctx, fuse):
    def ir_fusion(func):
        # pylint: disable=protected-access
        func = run_infer_type(func)
        func = mnm._ffi.pass_.FuseOps(func, 3)
        func = run_infer_type(func)
        return func

    def ir_identity(func):
        return func

    layers = [3, 4, 6, 3]
    ir_optimizer = ir_fusion if fuse else ir_identity
    m_model = MNMResNet50(layers)
    m_model.to(ctx=ctx)
    t_model = TorchResNet50(layers)
    t_model.to(ctx)
    init(m_model, t_model, layers, ctx=ctx)
    m_x, t_x = randn_torch([1, 3, 32, 32], requires_grad=True, ctx=ctx)
    m_y, t_y = one_hot(batch_size=1, num_classes=10, ctx=ctx)
    m_x.requires_grad = True
    m_model.train_mode()
    t_model.train()
    m_loss = run_vm_model(m_model, ctx, [m_x, m_y], ir_optimizer)[0]
    t_loss = t_model(t_x, t_y)
    check(m_loss, t_loss)
    check_params(m_model, t_model, layers)


if __name__ == "__main__":
    pytest.main([__file__])
