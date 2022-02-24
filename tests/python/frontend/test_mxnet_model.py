# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint:disable=invalid-name, superfluous-parens,too-many-locals,line-too-long
import pytest
from mxnet import gluon
import mxnet as mx
import gluoncv
import raf
from raf._op import sym
from raf.testing import randn_mxnet, one_hot_mxnet, get_testable_devices, check


param_map_list = []


def check_params(mx_model, raf_model):
    for param in param_map_list:
        mx_value = raf_model.state()[param]
        raf_value = mx_model.collect_params()[param].data()
        check(mx_value, raf_value, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "mx_model",
    [
        ["resnet18", gluon.model_zoo.vision.resnet18_v1(pretrained=True)],
        ["resnest14", gluoncv.model_zoo.get_model("resnest14", pretrained=True)],
    ],
)
@pytest.mark.parametrize("device", get_testable_devices())
def test_backward_check(device, mx_model):
    if device == "cpu" and mx_model[0] == "resnest14":
        pytest.skip(
            "skip since it contains op pooling which should be refactor to make schedule work"
        )
    mx_model[1].hybridize(static_alloc=True, static_shape=True)
    x, mx_x = randn_mxnet((5, 3, 224, 224), requires_grad=True, device=device)
    m_ytrue, mx_ytrue = one_hot_mxnet(batch_size=5, num_classes=1000, device=device)
    raf_model = raf.frontend.from_mxnet(mx_model[1], ["x"])

    out = raf_model.record(x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    raf_model = raf_model + loss

    for i in raf_model.state().keys():
        if i.find("running") == -1:
            param_map_list.append(i)

    raf_model.train_mode()
    raf_model.to(device=device)

    with mx.autograd.record():
        mx_out = mx_model[1](mx_x)
        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        mx_loss = softmax_loss(mx_out, mx_ytrue)
        mx_loss_v = mx_loss.mean().asscalar()
        mx_loss.backward()

    raf_loss_v = raf_model(x, m_ytrue)
    raf_loss_v.backward()
    check(mx_loss_v, raf_loss_v, rtol=1e-3, atol=1e-3)
    check_params(mx_model[1], raf_model)
    param_map_list.clear()


@pytest.mark.parametrize(
    "mx_model",
    [
        ["resnet18", gluon.model_zoo.vision.resnet18_v1(pretrained=True)],
        ["resnest14", gluoncv.model_zoo.get_model("resnest14", pretrained=True)],
    ],
)
@pytest.mark.parametrize("device", get_testable_devices())
def test_forward_check(device, mx_model):
    mx_model[1].hybridize(static_alloc=True, static_shape=True)

    x, mx_x = randn_mxnet((5, 3, 224, 224), requires_grad=True, device=device)
    m_ytrue, _ = one_hot_mxnet(batch_size=5, num_classes=1000, device=device)

    raf_model = raf.frontend.from_mxnet(mx_model[1], ["x"])
    raf_model.train_mode()
    raf_model.to(device=device)

    with mx.autograd.record():
        mx_model[1](mx_x)

    raf_model(x, m_ytrue)

    for i in raf_model.state().keys():
        if i.find("running") == -1:
            param_map_list.append(i)
    check_params(mx_model[1], raf_model)
    param_map_list.clear()


if __name__ == "__main__":
    pytest.main([__file__])
