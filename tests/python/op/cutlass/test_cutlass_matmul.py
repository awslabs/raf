# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-member
# pylint: disable=attribute-defined-outside-init
import pytest
import torch
import torch.nn.functional as F

import raf
from raf.testing import randn_torch, run_vm_model, check, DialectChecker
from raf.model.nn import Linear


def verify_ir(mod):
    with raf.device("cuda"):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        DialectChecker("cutlass").visit(mod["main"])


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("m", [1, 15])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("k", [16, 32])
@pytest.mark.parametrize(
    "epilogue",
    [
        [None, None],
        [raf._op.sym.relu, torch.nn.functional.relu],
        [raf._op.sym.gelu, torch.nn.GELU()],
    ],
)
@pytest.mark.parametrize("beta", [None, 0.5, 2.0])
def test_matmul_add_epilogue(m, n, k, epilogue, beta):
    m_epilogue, t_epilogue = epilogue

    class TestModel(raf.Model):
        def build(self):
            self.epilogue = m_epilogue
            if beta:
                self.beta = raf.array(beta, dtype="float32")

        @raf.model.trace
        def forward(self, x, w, bias):  # pylint: disable=no-self-use
            x = raf.matmul(x, w)
            scaled_bias = raf.multiply(self.beta, bias) if beta else bias
            x = raf.add(x, scaled_bias)
            x = self.epilogue(x) if self.epilogue else x
            return x

    device = "cuda"
    m_x, t_x = randn_torch([m, k], requires_grad=True, device=device)
    m_w, t_w = randn_torch([k, n], requires_grad=True, device=device)
    m_bias, t_bias = randn_torch(
        [
            n,
        ],
        requires_grad=True,
        device=device,
    )
    model = TestModel()
    model.to(device=device)
    mod = model._internal(m_x, m_w, m_bias).mod
    verify_ir(mod)
    m_y = run_vm_model(model, device, [m_x, m_w, m_bias])
    t_scaled_bias = t_bias * beta if beta else t_bias
    t_y = torch.matmul(t_x, t_w) + t_scaled_bias
    t_y = t_epilogue(t_y) if t_epilogue else t_y
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("in_features", [16, 32])
@pytest.mark.parametrize("out_features", [16, 32])
def test_dense_add_relu(batch_size, in_features, out_features):
    class TestModel(raf.Model):
        def build(self, in_features, out_features):
            self.linear = Linear(in_features, out_features, bias=True)

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            x = self.linear(x)
            x = raf.relu(x)
            return x

    device = "cuda"
    m_x, t_x = randn_torch([batch_size, in_features], requires_grad=True, device=device)
    m_w, _ = randn_torch([out_features, in_features], requires_grad=True, device=device)
    m_b, _ = randn_torch([out_features], requires_grad=True, device=device)
    t_model = torch.nn.Linear(in_features, out_features, bias=True)
    t_model.weight.data[:] = torch.from_numpy(m_w.numpy())
    t_model.bias.data[:] = torch.from_numpy(m_b.numpy())
    model = TestModel(in_features=in_features, out_features=out_features)
    model.linear.w = m_w
    model.linear.b = m_b
    model.to(device=device)
    t_model.to(device=device)
    mod = model._internal(m_x).mod
    verify_ir(mod)
    m_y = run_vm_model(model, device, [m_x])
    t_y = F.relu(t_model(t_x))
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("batch_size1", [1, 15])
@pytest.mark.parametrize("batch_size2", [1, 15])
@pytest.mark.parametrize("m", [2, 12])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("k", [16, 32])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("epilogue", [[None, None], [raf._op.sym.gelu, torch.nn.GELU()]])
def test_batch_matmul_nt_add(batch_size1, batch_size2, m, n, k, dtype, epilogue):
    m_epilogue, t_epilogue = epilogue

    class TestModel(raf.Model):
        def build(self):
            self.epilogue = m_epilogue

        @raf.model.trace
        def forward(self, x, w, bias):  # pylint: disable=no-self-use
            x = raf.batch_matmul_nt(x, w)
            x = raf.add(x, bias)
            x = self.epilogue(x) if self.epilogue else x
            return x

    device = "cuda"
    batch_size = max(batch_size1, batch_size2)
    m_x, t_x = randn_torch([batch_size1, m, k], device=device, dtype=dtype)
    m_w, t_w = randn_torch([batch_size2, n, k], device=device, dtype=dtype)
    m_b, t_b = randn_torch([batch_size, m, n], device=device, dtype=dtype)
    model = TestModel()
    model.to(device=device, dtype=dtype)
    mod = model._internal(m_x, m_w, m_b).mod
    verify_ir(mod)
    m_y = run_vm_model(model, device, [m_x, m_w, m_b])
    t_y = torch.matmul(t_x, t_w.permute(0, 2, 1)) + t_b
    t_y = t_epilogue(t_y) if t_epilogue else t_y
    atol = rtol = 1e-4 if dtype == "float32" else 1e-1
    check(m_y, t_y, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
