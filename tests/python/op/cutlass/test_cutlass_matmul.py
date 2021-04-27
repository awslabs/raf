# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-member
# pylint: disable=attribute-defined-outside-init
import pytest
import torch
import torch.nn.functional as F

import mnm
from mnm.testing import randn_torch, run_vm_model, check
from mnm.model import Linear

@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("m", [1, 15])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("k", [16, 32])
def test_matmul_add_relu(m, n, k):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w, bias):  # pylint: disable=no-self-use
            x = mnm.matmul(x, w)
            x = mnm.add(x, bias)
            x = mnm.relu(x)
            return x

    device = "cuda"
    m_x, t_x = randn_torch([m, k], requires_grad=True, device=device)
    m_w, t_w = randn_torch([k, n], requires_grad=True, device=device)
    m_bias, t_bias = randn_torch([n,], requires_grad=True, device=device)
    model = TestModel()
    model.to(device=device)
    m_y = run_vm_model(model, device, [m_x, m_w, m_bias], mnm._ffi.pass_.FuseOps(3))
    t_y = torch.nn.functional.relu(torch.matmul(t_x, t_w) + t_bias)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("in_features", [16, 32])
@pytest.mark.parametrize("out_features", [16, 32])
def test_dense_add_relu(batch_size, in_features, out_features):
    class TestModel(mnm.Model):
        def build(self, in_features, out_features):
            self.linear = Linear(in_features, out_features, bias=True)

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            x = self.linear(x)
            x = mnm.relu(x)
            return x

    device = "cuda"
    m_x, t_x = randn_torch([batch_size, in_features], requires_grad=True, device=device)
    m_w, _ = randn_torch([out_features, in_features], requires_grad=True, device=device)
    m_b, _ = randn_torch([out_features], requires_grad=True, device=device)
    t_model = torch.nn.Linear(in_features, out_features, bias=True)
    t_model.weight[:] = torch.from_numpy(m_w.asnumpy())
    t_model.bias[:] = torch.from_numpy(m_b.asnumpy())
    model = TestModel(in_features=in_features,
                      out_features=out_features)
    model.linear.w = m_w
    model.linear.b = m_b
    model.to(device=device)
    t_model.to(device=device)
    m_y = run_vm_model(model, device, [m_x], mnm._ffi.pass_.FuseOps(3))
    t_y = F.relu(t_model(t_x))
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("batch_size1", [1, 15])
@pytest.mark.parametrize("batch_size2", [1, 15])
@pytest.mark.parametrize("m", [2, 12])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("k", [16, 32])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_batch_matmul_nt_add(batch_size1, batch_size2, m, n, k, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w, bias):  # pylint: disable=no-self-use
            x = mnm.batch_matmul_nt(x, w)
            x = mnm.add(x, bias)
            return x

    device = "cuda"
    batch_size = max(batch_size1, batch_size2)
    m_x, t_x = randn_torch([batch_size1, m, k], device=device, dtype=dtype)
    m_w, t_w = randn_torch([batch_size2, n, k], device=device, dtype=dtype)
    m_b, t_b = randn_torch([batch_size, m, n], device=device, dtype=dtype)
    model = TestModel()
    model.to(device=device)
    m_y = run_vm_model(model, device, [m_x, m_w, m_b], mnm._ffi.pass_.FuseOps(3))
    t_y = torch.matmul(t_x, t_w.permute(0, 2, 1)) + t_b
    atol = rtol = 1e-4 if dtype == "float32" else 1e-1
    check(m_y, t_y, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
