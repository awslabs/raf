# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint:disable=invalid-name, superfluous-parens,too-many-locals,line-too-long
import pytest
from mxnet import gluon
import mxnet as mx
import gluoncv
import mnm
from mnm._op import sym
from mnm.testing import randn_mxnet, one_hot_mxnet, get_testable_devices, check


param_map_list = []


def check_params(mx_model, mnm_model):
    for param in param_map_list:
        mx_value = mnm_model.state()[param]
        mnm_value = mx_model.collect_params()[param].data()
        check(mx_value, mnm_value, rtol=1e-3, atol=1e-3)


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
    mnm_model = mnm.frontend.from_mxnet(mx_model[1], ["x"])

    out = mnm_model.record(x)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(m_ytrue, y_pred)
    mnm_model = mnm_model + loss

    for i in mnm_model.state().keys():
        if i.find("running") == -1:
            param_map_list.append(i)

    mnm_model.train_mode()
    mnm_model.to(device=device)

    with mx.autograd.record():
        mx_out = mx_model[1](mx_x)
        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        mx_loss = softmax_loss(mx_out, mx_ytrue)
        mx_loss_v = mx_loss.mean().asscalar()
        mx_loss.backward()

    mnm_loss_v = mnm_model(x, m_ytrue)
    mnm_loss_v.backward()
    check(mx_loss_v, mnm_loss_v, rtol=1e-3, atol=1e-3)
    check_params(mx_model[1], mnm_model)
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

    mnm_model = mnm.frontend.from_mxnet(mx_model[1], ["x"])
    mnm_model.train_mode()
    mnm_model.to(device=device)

    with mx.autograd.record():
        mx_model[1](mx_x)

    mnm_model(x, m_ytrue)

    for i in mnm_model.state().keys():
        if i.find("running") == -1:
            param_map_list.append(i)
    check_params(mx_model[1], mnm_model)
    param_map_list.clear()


if __name__ == "__main__":
    pytest.main([__file__])
