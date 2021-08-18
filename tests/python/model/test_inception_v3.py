# pylint: disable=protected-access
import pytest
import mnm
from mnm.testing import check, run_vm_model, get_device_list, inception


@pytest.mark.parametrize("block", ["A"])
@pytest.mark.parametrize("device", get_device_list())
def test_block_intpr_forward(block, device):
    (m_model, m_x, m_y), (t_model, t_x, t_y) = inception.get_block_and_input(block, device=device)
    m_loss = m_model(m_x, m_y)
    t_loss = t_model(t_x, t_y)
    m_loss.backward()
    t_loss.backward()
    check(m_loss, t_loss, rtol=1e-3, atol=1e-3)
    inception.check_params(m_model, t_model, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block", ["A"])
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("fuse_lv", [0, 1])
def test_block_vm_forward(block, device, fuse_lv):
    (m_model, m_x, m_y), (t_model, t_x, t_y) = inception.get_block_and_input(block, device=device)
    m_loss = run_vm_model(m_model, device, [m_x, m_y], fuse_level=fuse_lv)[0]
    t_loss = t_model(t_x, t_y)
    check(m_loss, t_loss, rtol=1e-3, atol=1e-3)
    inception.check_params(m_model, t_model, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_inception_v3_intpr_forward(device='cuda'):
    m_model, t_model = inception.get_model()
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = inception.get_input(batch_size=1, device=device)
    m_loss = m_model(*m_in)
    t_loss = t_model(*t_in)
    m_loss.backward()
    t_loss.backward()
    check(m_loss, t_loss, rtol=1e-3, atol=1e-3)
    inception.check_params(m_model, t_model, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse_lv", [0, 1])
def test_vm_forward(fuse_lv):
    device = 'cuda'
    m_model, t_model = inception.get_model()
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = inception.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_model, device, [*m_in], fuse_level=fuse_lv)[0]
    t_loss = t_model(*t_in)
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    inception.check_params(m_model, t_model, atol=1e-3, rtol=1e-3)

#
# TODO(yaoyaoding): Add vm backward test after the accuracy issue is resolved.
#


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("block_name", ["a", "ab"])
@pytest.mark.parametrize("fuse_lv", [0, 1])
@pytest.mark.parametrize("stream_schedule_policy", ["wavefront", "asap"])
def test_block_vm_multi_stream(block_name, stream_schedule_policy, fuse_lv):
    if block_name == "ab":
        pytest.xfail(reason="Workspace of kernels on different stream may overlap.")
    device = 'cuda'
    (model, x, _), _ = inception.get_block_and_input(block_name=block_name, device=device)
    model.infer_mode()
    y_1 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy='sequential')
    y_2 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy=stream_schedule_policy)
    check(y_1, y_2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.xfail(reason="Workspace of kernels on different stream may overlap.")
@pytest.mark.parametrize("fuse_lv", [0, 1])
@pytest.mark.parametrize("stream_schedule_policy", ["wavefront", "asap"])
def test_vm_multi_stream(stream_schedule_policy, fuse_lv):
    device = 'cuda'
    model, _ = inception.get_model()
    model.to(device=device)
    model.infer_mode()
    (x, _), _ = inception.get_input(batch_size=1, device=device)
    y_1 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy='sequential')
    y_2 = run_vm_model(model, device, [x], fuse_lv, stream_schedule_policy=stream_schedule_policy)
    check(y_1, y_2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
