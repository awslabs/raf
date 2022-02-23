# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,no-member,protected-access
import pytest
import numpy as np

import mnm
from mnm.testing import check, run_vm_model, randn, get_testable_devices


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("device", get_testable_devices())
def test_size(shape, axis, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self, axis):
            self.axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.size(x, self.axis)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.size(m_x, axis)
    n_y = np.array(n_x.shape[axis], dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model(axis=axis)
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_numel(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.numel(x)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.numel(m_x)
    n_y = np.array(n_x.size, dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model()
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


@pytest.mark.parametrize("shape", [[5, 3], [5, 3, 2], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_shape_as_tensor(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.shape_as_tensor(x)

    m_x, n_x = randn(shape, device=device)
    # imperative
    m_y = mnm.shape_as_tensor(m_x)
    n_y = np.array(n_x.shape, dtype="int32")
    assert m_y.shape == n_y.shape
    assert (m_y.numpy() == n_y).all()
    # traced
    if device != "cuda":  # TODO: vm not support heterogeneous now
        model = Model()
        v_y = run_vm_model(model, device, [m_x], opt_level=1)
        assert v_y.shape == n_y.shape
        assert (v_y.numpy() == n_y).all()


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_device_copy():
    shape = (4, 4)

    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data):
            a_1 = mnm.exp(data)
            a_2 = mnm.device_copy(a_1, "cuda", "cpu")
            a_3 = mnm.exp(a_1)
            a_4 = mnm.device_copy(a_2, "cpu", "cuda")
            return mnm.add(a_3, a_4)

    data, data_np = randn(shape, device="cuda", positive=True)

    model = Model()
    out = run_vm_model(model, "cuda", [data], opt_level=3)
    ref_out = np.exp(data_np) + np.exp(np.exp(data_np))
    check(out, ref_out)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_async_device_copy():
    shape = (4, 4)
    comp_stream = 1
    copy_stream = 2

    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data):
            mnm.set_stream(0, comp_stream)
            a_1 = mnm.exp(data)
            mnm.set_stream(0, copy_stream)
            a_2 = mnm.device_copy(a_1, "cuda", "cuda_host")
            mnm.add_event(100, copy_stream)
            mnm.set_stream(0, comp_stream)
            a_3 = mnm.exp(a_1)
            mnm.set_stream(0, copy_stream)
            mnm.wait_event(100, copy_stream)
            a_4 = mnm.device_copy(a_2, "cuda_host", "cuda")
            mnm.add_event(101, copy_stream)
            mnm.set_stream(0, comp_stream)
            mnm.wait_event(101, comp_stream)
            return mnm.add(a_3, a_4)

    data, data_np = randn(shape, device="cuda", positive=True)

    model = Model()
    # Disable GNF/BBNF to preserve set_stream/add_event/wait_event and their orders.
    # This is only for testing.
    out = run_vm_model(model, "cuda", [data], opt_level=3, anf_only=True)
    ref_out = np.exp(data_np) + np.exp(np.exp(data_np))
    check(out, ref_out)


if __name__ == "__main__":
    pytest.main([__file__])
