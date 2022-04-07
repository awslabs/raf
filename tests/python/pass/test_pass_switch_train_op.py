# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
import pytest
import raf
from raf._ffi.pass_ import InferType, SwitchTrainOp
from raf.testing import randn_torch, check, run_vm_model, get_testable_devices


@pytest.mark.parametrize("device", get_testable_devices())
def test_layer_norm(device):
    shape = (3, 2, 4)
    scale_shape = [shape[-1]]
    dtype = "float32"

    class Model(raf.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @raf.model.trace
        def forward(self, x, scale, bias):
            out = raf.relu(x)
            out = raf.layer_norm(out, scale, bias, axis=self._axis, eps=self._eps)
            return raf.relu(out)

    m_model = Model(-1, 1e-12)
    m_model.to(device=device, dtype=dtype)

    m_x, _ = randn_torch(shape, device=device, dtype=dtype, requires_grad=True)
    m_scale, _ = randn_torch(scale_shape, device=device, dtype=dtype, requires_grad=True)
    m_bias, _ = randn_torch(scale_shape, device=device, dtype=dtype, requires_grad=True)
    args = [m_x, m_scale, m_bias]

    out_ref = run_vm_model(m_model, device, args)
    mod = InferType()(m_model._internal(*args).mod)

    train_mod = SwitchTrainOp(True)(mod)
    infer_mod = SwitchTrainOp(False)(mod)
    model = raf.frontend.FrameworkModel(train_mod, infer_mod, {}, {})

    model.train_mode()
    assert raf.ir.AsText(train_mod).find("layer_norm_train") != -1
    out_switch = run_vm_model(model, device, args)
    check(out_ref, out_switch)

    model.infer_mode()
    assert raf.ir.AsText(infer_mod).find("layer_norm_train") == -1
    out_switch = run_vm_model(model, device, args)
    check(out_ref, out_switch)


if __name__ == "__main__":
    pytest.main([__file__])
