# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
import numpy as np
import pytest
import mnm
from mnm.testing import randn, check, run_vm_model


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("d_type", ["float64", "float32", "float16"])
def test_batch_matmul(b, n, m, k, d_type, device, transpose_a=False, transpose_b=True):
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):
            mnm_op = mnm.batch_matmul
            return mnm_op(m_a, m_b)
    # forward
    model = TestModel()
    m_a, n_a = randn((b, n, k) if not transpose_a else (b, k, n), dtype=d_type, device=device)
    m_b, n_b = randn((b, k, m) if not transpose_b else (b, m, k), dtype=d_type, device=device)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b])
    t_c = np.matmul(n_a, np.transpose(n_b, (0, 2, 1))) # pylint: disable=no-member
    rtol = 4e-2 if d_type == "float16" else 1e-4
    atol = 4e-2 if d_type == "float16" else 1e-4

    check(m_c, t_c, rtol=rtol, atol=atol)
    check(v_c, t_c, rtol=rtol, atol=atol)
    # backward
    m_dy, n_dy = randn(m_c.shape, dtype=d_type, device=device)
    m_c.backward(m_dy)
    n_dyt = np.transpose(n_dy, (0, 2, 1))
    check(m_a.grad, np.matmul(n_dy, n_b), rtol=rtol, atol=atol)
    check(m_b.grad, np.matmul(n_dyt, n_a), rtol=rtol, atol=atol)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("d_type", ["float64", "float32", "float16"])
@pytest.mark.parametrize("broadcast_a", [True, False])
def test_batch_matmul_broadcast(b, n, m, k, broadcast_a, d_type, device, transpose_a=False, transpose_b=True):
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, m_a, m_b):
            mnm_op = mnm.batch_matmul
            print("forward once")
            return mnm_op(m_a, m_b)
    # forward
    model = TestModel()
    if(broadcast_a):
        m_a, n_a = randn((1, n, k) if not transpose_a else (1, k, n), dtype=d_type, device=device)
        m_b, n_b = randn((b, k, m) if not transpose_b else (b, m, k), dtype=d_type, device=device)
    else:
        m_a, n_a = randn((b, n, k) if not transpose_a else (b, k, n), dtype=d_type, device=device)
        m_b, n_b = randn((1, k, m) if not transpose_b else (1, m, k), dtype=d_type, device=device)
    m_a.requires_grad = True
    m_b.requires_grad = True
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b])
    t_c = np.matmul(n_a, np.transpose(n_b, (0, 2, 1))) # pylint: disable=no-member
    rtol = 4e-2 if d_type == "float16" else 1e-4
    atol = 4e-2 if d_type == "float16" else 1e-4
    check(m_c, t_c, rtol=rtol, atol=atol)
    check(v_c, t_c, rtol=rtol, atol=atol)
    # backward
    m_dy, n_dy = randn(m_c.shape, dtype=d_type, device=device)
    m_c.backward(m_dy)
    n_dyt = np.transpose(n_dy, (0, 2, 1))
    if(broadcast_a):
        check(m_a.grad, np.sum(np.matmul(n_dy, n_b), axis=0, keepdims=True), rtol=rtol, atol=atol)
        check(m_b.grad, np.matmul(n_dyt, n_a), rtol=rtol, atol=atol)
    else:
        check(m_a.grad, np.matmul(n_dy, n_b), rtol=rtol, atol=atol)
        check(m_b.grad, np.sum(np.matmul(n_dyt, n_a), axis=0, keepdims=True), rtol=rtol, atol=atol)

if __name__ == "__main__":
    pytest.main([__file__])
