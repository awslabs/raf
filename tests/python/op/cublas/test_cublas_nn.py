# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
import pytest
import torch
import mnm
from mnm.testing import check, randn_torch, run_vm_model

@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("broadcast", ["none", "a", "b"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_batch_matmul(device, dtype, b, n, k, m, broadcast, transpose_a, transpose_b):
    # pylint: disable=too-many-arguments, invalid-name
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):
            mnm_op = [[mnm.batch_matmul, mnm.batch_matmul_nt],
                      [mnm.batch_matmul_tn, mnm.batch_matmul_tt]]
            mnm_op = mnm_op[transpose_a][transpose_b]
            return mnm_op(m_a, m_b)

    b1 = b
    b2 = b
    if broadcast == "a":
        b1 = 1
    elif broadcast == "b":
        b2 = 1
    # forward
    model = TestModel()
    m_a, t_a = randn_torch((b1, n, k) if not transpose_a else (b1, k, n),
                           device=device, dtype=dtype, requires_grad=True)
    m_b, t_b = randn_torch((b2, k, m) if not transpose_b else (b2, m, k),
                           device=device, dtype=dtype, requires_grad=True)
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b])

    t_at = torch.transpose(t_a, 1, 2) if transpose_a else t_a
    t_bt = torch.transpose(t_b, 1, 2) if transpose_b else t_b
    t_c = torch.matmul(t_at, t_bt) # pylint: disable=no-member

    tol = 1e-4
    if dtype == "float16":
        tol = 5e-2
    check(m_c, t_c, rtol=tol, atol=tol)
    check(v_c, t_c, rtol=tol, atol=tol)
    # backward
    m_dc, t_dc = randn_torch(m_c.shape, device=device, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad, rtol=tol, atol=tol)
    check(m_b.grad, t_b.grad, rtol=tol, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__])
