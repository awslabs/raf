# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
from raf.testing import check, run_vm_model, get_testable_devices, inception


@pytest.mark.parametrize("block", ["A"])
@pytest.mark.parametrize("device", get_testable_devices())
def test_block_intpr_forward(block, device):
    (m_model, m_x, m_y), (t_model, t_x, t_y) = inception.get_block_and_input(block, device=device)
    m_loss = m_model(m_x, m_y)
    t_loss = t_model(t_x, t_y)
    m_loss.backward()
    t_loss.backward()
    check(m_loss, t_loss, rtol=1e-3, atol=1e-3)
    inception.check_params(m_model, t_model, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block", ["A"])
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("fuse", [False, True])
def test_block_vm_forward(block, device, fuse):
    (m_model, m_x, m_y), (t_model, t_x, t_y) = inception.get_block_and_input(block, device=device)
    m_loss = run_vm_model(m_model, device, [m_x, m_y], disable_fusion=not fuse)[0]
    t_loss = t_model(t_x, t_y)
    check(m_loss, t_loss, rtol=1e-3, atol=1e-3)
    inception.check_params(m_model, t_model, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_inception_v3_intpr_forward(device="cuda"):
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


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("fuse", [False, True])
def test_vm_forward(fuse):
    device = "cuda"
    m_model, t_model = inception.get_model()
    m_model.to(device=device)
    t_model.to(device)
    m_in, t_in = inception.get_input(batch_size=1, device=device)
    m_loss = run_vm_model(m_model, device, [*m_in], disable_fusion=not fuse)[0]
    t_loss = t_model(*t_in)
    check(m_loss, t_loss, atol=1e-3, rtol=1e-3)
    inception.check_params(m_model, t_model, atol=1e-3, rtol=1e-3)


#
# TODO(yaoyaoding): Add vm backward test after the accuracy issue is resolved.
#


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("block_name", ["c"])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("policy", ["wavefront", "asap"])
def test_block_vm_multi_stream(block_name, policy, fuse):
    device = "cuda"
    (model, x, _), _ = inception.get_block_and_input(block_name=block_name, device=device)
    model.infer_mode()

    y_1 = run_vm_model(
        model, device, [x], disable_fusion=not fuse, stream_schedule_policy="sequential"
    )
    y_2 = run_vm_model(model, device, [x], disable_fusion=not fuse, stream_schedule_policy=policy)
    check(y_1, y_2, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(True, reason="Skip to save the CI time")
@pytest.mark.skipif(
    raf.build.with_cuda() and float(raf.build.with_cuda()) <= 11.2,
    reason="Workspace may overlap for cuda <= 11.2.",
)
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("policy", ["wavefront", "asap", "ios"])
def test_vm_multi_stream(policy, fuse):
    device = "cuda"
    model, _ = inception.get_model()
    model.to(device=device)
    model.infer_mode()
    (x, _), _ = inception.get_input(batch_size=1, device=device)
    for _ in range(2):
        y_1 = run_vm_model(
            model, device, [x], disable_fusion=not fuse, stream_schedule_policy="sequential"
        )
        y_2 = run_vm_model(
            model, device, [x], disable_fusion=not fuse, stream_schedule_policy=policy
        )
        check(y_1, y_2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
