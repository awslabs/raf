# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import raf
from tvm import relay
from raf.testing import (
    randn_torch,
    get_testable_devices,
    check,
    run_vm_model,
    get_vm_executor,
    resnet,
    mlp,
)
from raf._core.ndarray import Symbol
from raf.model.trace import _get_func_inputs


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("fuse", [True, False])
def test_simple1(device, fuse):
    # pylint: disable=no-self-use
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.argwhere(x)
            y = raf.split(y, 2)
            y = raf.add(y[0], y[1])
            y = raf.abs(y)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = raf.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x], disable_fusion=not fuse)
    expected = raf.array([[1, 0], [1, 2]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("fuse", [True, False])
def test_simple2(device, fuse):
    # pylint: disable=no-self-use
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.argwhere(x)
            y = raf.split(y, 2)
            y = raf.add(y[0], y[1])
            y = raf.sum(y)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = raf.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x], disable_fusion=not fuse)
    expected = raf.array(4, dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", get_testable_devices())
def test_dynamic_reshape(device):
    # pylint: disable=no-self-use
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.argwhere(x)
            y = raf.split(y, 2)
            y = raf.add(y[0], y[1])
            y = raf.abs(y)
            y = raf.expand_dims(y, 0)
            return y

    model = Model()
    n_x = np.ones((2, 2)).astype("float32")
    m_x = raf.array(n_x, device=device)
    m_res = model(m_x)
    v_res = run_vm_model(model, device, [m_x])
    expected = raf.array([[[1, 0], [1, 2]]], dtype="int32")
    check(m_res, expected)
    check(v_res, expected)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("fuse", [True, False])
def test_resnet_forward(device, fuse):
    # pylint: disable=invalid-name, protected-access
    m_model, _ = resnet.get_model([1, 1, 1, 1], False)
    m_model.to(device=device)

    x_ty = relay.TensorType((relay.Any(), 3, 224, 224))
    x = Symbol.make_var("x", x_ty)
    record = m_model._internal(x)
    mod = record.mod
    vm = get_vm_executor(mod, device, 2, not fuse)

    (m_x, _), _ = resnet.get_input(batch_size=1, device=device)
    m_x.requires_grad = False
    inputs = _get_func_inputs(record, (m_x,), {}, get_handle=False)

    v_res = vm(*inputs)
    m_res = m_model(m_x)
    check(m_res, v_res)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("fuse", [True, False])
def test_resnet_backward(device, fuse):
    # pylint: disable=invalid-name, protected-access, too-many-locals
    m_model, t_model = resnet.get_model([1, 1, 1, 1])
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)

    x_ty = relay.TensorType((relay.Any(), 3, 224, 224))
    x = Symbol.make_var("x", x_ty)
    yhat_ty = relay.TensorType((relay.Any(),))
    yhat = Symbol.make_var("yhat", yhat_ty)
    dy_ty = relay.TensorType(())
    dy = Symbol.make_var("dy", dy_ty)
    record = m_optimizer._internal(dy, x, yhat)
    mod = record.mod

    m_dy, t_dy = randn_torch((), device=device, requires_grad=True)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    vm = get_vm_executor(mod, device, 2, not fuse)
    inputs = _get_func_inputs(record, (m_dy, *m_in), {}, get_handle=False)
    m_loss = vm(*inputs)[0][0]

    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()

    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    resnet.check_params(m_model, t_model, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("config", [(784, 10, 256, 256)])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("fuse", [True, False])
def test_mlp(config, device, fuse):
    # pylint: disable=invalid-name, protected-access, too-many-locals
    m_model, t_model = mlp.get_model(config)
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)

    x_ty = relay.TensorType((relay.Any(), config[0]))
    x = Symbol.make_var("x", x_ty)
    yhat_ty = relay.TensorType((relay.Any(),))
    yhat = Symbol.make_var("yhat", yhat_ty)
    dy_ty = relay.TensorType(())
    dy = Symbol.make_var("dy", dy_ty)
    record = m_optimizer._internal(dy, x, yhat)
    mod = record.mod

    m_dy, t_dy = randn_torch((), device=device, requires_grad=True)
    m_in, t_in = mlp.get_input(config, batch_size=1, device=device)
    vm = get_vm_executor(mod, device, 2, not fuse)
    inputs = _get_func_inputs(record, (m_dy, *m_in), {}, get_handle=False)
    m_loss = vm(*inputs)[0]

    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()

    check(m_loss, t_loss, atol=1e-4, rtol=1e-4)
    mlp.check_params(m_model, t_model)


if __name__ == "__main__":
    pytest.main([__file__])
