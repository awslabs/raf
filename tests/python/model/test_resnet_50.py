# pylint: disable=protected-access
import pytest
import torch

import mnm
from mnm.testing import check, randn_torch, run_vm_model, resnet


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_r50_v1_imagenet():
    device = "cuda"
    m_model, t_model = resnet.get_model([3, 4, 6, 3])
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = m_model(*m_in)
    t_loss = t_model(*t_in)
    m_loss.backward()
    t_loss.backward()
    check(m_loss, t_loss)
    resnet.check_params(m_model, t_model)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse_lv", [0, 1])
def test_vm_forward(fuse_lv):
    device = "cuda"
    layers = [3, 4, 6, 3]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_model, device, [*m_in], fuse_level=fuse_lv)[0]
    t_loss = t_model(*t_in)
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    resnet.check_params(m_model, t_model, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse_lv", [0, 1])
def test_vm_backward(fuse_lv):
    device = "cuda"
    layers = [1, 1, 1, 1]
    m_model, t_model = resnet.get_model(layers)
    m_model.to(device=device)
    t_model.to(device=device)
    m_optimizer = mnm.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(m_model)
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.1, momentum=0.01)
    m_dy, t_dy = randn_torch((), device=device, requires_grad=False)
    m_in, t_in = resnet.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_optimizer, device, [m_dy, *m_in], fuse_level=fuse_lv)[0][0]
    t_optimizer.zero_grad()
    t_loss = t_model(*t_in)
    t_loss.backward(t_dy)
    t_optimizer.step()
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    resnet.check_params(m_model, t_model, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse_lv", [0, 1])
@pytest.mark.parametrize("stream_schedule_policy", ["wavefront", "asap"])
def test_vm_multi_stream(stream_schedule_policy, fuse_lv):
    device = 'cuda'
    layers = [3, 4, 6, 3]
    model, _ = resnet.get_model(layers)
    model.to(device=device)
    model.infer_mode()
    (x, _), _ = resnet.get_input(batch_size=1, device=device)
    y_1 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy='sequential')
    y_2 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy=stream_schedule_policy)
    check(y_1, y_2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
