# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import numpy as np
import torch

import raf
from raf.testing import check, randn_torch, run_vm_model, resnet, with_seed


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_interpreter_fp32():
    x = np.random.randn(1, 3, 32, 32)
    y = np.random.randn(
        1,
    )
    m_x = raf.array(x, dtype="float32", device="cuda")
    m_y = raf.array(y, dtype="int64", device="cuda")
    model = resnet.RAFResNet50([3, 4, 6, 3])
    model.to(device="cuda")
    model.train_mode()
    model(m_x, m_y)
    model.infer_mode()
    model(m_x)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse", [False, True])
@with_seed(0)
def test_vm_forward(fuse):
    device = "cuda"
    layers = [3, 4, 6, 3]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_model, device, [*m_in], disable_fusion=not fuse)[0]
    t_loss = t_model(*t_in)
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    resnet.check_params(m_model, t_model, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse", [False, True])
@with_seed(0)
def test_vm_backward(fuse):
    device = "cuda"
    layers = [1, 1, 1, 1]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    m_dy, t_dy = randn_torch((), device=device, requires_grad=False)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_optimizer, device, [m_dy, *m_in], disable_fusion=not fuse)[0][0]
    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    resnet.check_params(m_model, t_model, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse", [True])
@pytest.mark.parametrize("policy", ["wavefront", "asap", "ios"])
@with_seed(0)
def test_vm_multi_stream(policy, fuse):
    device = "cuda"
    layers = [3, 4, 6, 3]
    model, _ = resnet.get_model(layers)
    model.to(device=device)
    model.infer_mode()
    (x, _), _ = resnet.get_input(batch_size=1, device=device)
    y_1 = run_vm_model(
        model, device, [x], disable_fusion=not fuse, stream_schedule_policy="sequential"
    )
    y_2 = run_vm_model(model, device, [x], disable_fusion=not fuse, stream_schedule_policy=policy)
    check(y_1, y_2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
