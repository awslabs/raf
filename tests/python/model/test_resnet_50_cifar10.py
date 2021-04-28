# pylint: disable=protected-access
import pytest
import numpy as np
import torch

import mnm
from mnm.testing import resnet_cifar10 as resnet
from mnm.testing import run_vm_model, check, randn_torch, get_device_list, with_seed


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_build():
    x = np.random.randn(1, 3, 32, 32)
    y = np.random.randn(1, 10)
    m_x = mnm.array(x, dtype="float32", device="cuda")
    m_y = mnm.array(y, dtype='float32', device='cuda')
    model = resnet.MNMResNet50([3, 4, 6, 3])
    model.to(device='cuda')
    print("### Switch to training mode")
    model.train_mode()
    model(m_x, m_y)
    print("### Switch to infer mode")
    model.infer_mode()
    model(m_x)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@with_seed(0)
def test_build_fp16():
    x = np.random.randn(1, 3, 32, 32)
    m_x = mnm.array(x, dtype="float32", device="cuda")
    model = resnet.MNMResNet50([3, 4, 6, 3])
    model.to(device='cuda')
    model.infer_mode()
    m_y1 = model(m_x)
    print("### Switch to AMP model")
    amp_model = mnm.amp.autocast(model)
    m_y2 = amp_model(m_x)
    np.testing.assert_allclose(m_y1.asnumpy(), m_y2.asnumpy(), rtol=0.1, atol=0.1)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fuse_lv", [0, 1])
@with_seed(0)
def test_vm_forward(device, fuse_lv):
    layers = [3, 4, 6, 3]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device=device)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_model, device, [*m_in], mnm._ffi.pass_.FuseOps(fuse_lv))[0]
    t_loss = t_model(*t_in)
    check(m_loss, t_loss, rtol=1e-4, atol=1e-4)
    resnet.check_params(m_model, t_model)



@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fuse_lv", [0, 1])
@with_seed(0)
def test_vm_backward(device, fuse_lv):
    layers = [1, 1, 1, 1]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = mnm.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    m_dy, t_dy = randn_torch((), device=device, requires_grad=False)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_optimizer, device, [m_dy, *m_in], mnm._ffi.pass_.FuseOps(fuse_lv))[0][0]
    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()
    check(m_loss, t_loss)
    resnet.check_params(m_model, t_model)


if __name__ == "__main__":
    pytest.main([__file__])
