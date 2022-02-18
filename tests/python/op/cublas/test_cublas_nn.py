# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
import pytest
import torch
import mnm
from mnm.testing import check, randn_torch, run_vm_model, with_seed


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mnm_matmul(n, k, m, transpose_a, transpose_b, dtype):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_a, m_b):
            mnm_op = [[mnm.matmul, mnm.matmul_nt], [mnm.matmul_tn, mnm.matmul_tt]]
            mnm_op = mnm_op[transpose_a][transpose_b]
            return mnm_op(m_a, m_b)

    # forward
    model = TestModel()
    m_a, t_a = randn_torch(
        (n, k) if not transpose_a else (k, n), dtype=dtype, device="cuda", requires_grad=True
    )
    m_b, t_b = randn_torch(
        (k, m) if not transpose_b else (m, k), dtype=dtype, device="cuda", requires_grad=True
    )
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, "cuda", [m_a, m_b])
    t_c = torch.matmul(t_a.T if transpose_a else t_a, t_b.T if transpose_b else t_b)
    check(m_c, t_c)
    check(v_c, t_c)

    # backward
    m_dc, t_dc = randn_torch(m_c.shape, dtype=dtype, device="cuda")
    m_c.backward(m_dc)
    t_c.backward(t_dc)
    check(m_a.grad, t_a.grad)
    check(m_b.grad, t_b.grad)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("b", [2, 4])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("broadcast", ["none", "a", "b"])
@pytest.mark.parametrize("transpose_a", [True, False])
@pytest.mark.parametrize("transpose_b", [True, False])
@with_seed(0)
def test_batch_matmul(dtype, b, n, k, m, broadcast, transpose_a, transpose_b):
    # pylint: disable=too-many-arguments, invalid-name
    device = "cuda"

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_a, m_b):
            mnm_op = [
                [mnm.batch_matmul, mnm.batch_matmul_nt],
                [mnm.batch_matmul_tn, mnm.batch_matmul_tt],
            ]
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
    m_a, t_a = randn_torch(
        (b1, n, k) if not transpose_a else (b1, k, n),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    m_b, t_b = randn_torch(
        (b2, k, m) if not transpose_b else (b2, m, k),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    m_c = model(m_a, m_b)
    v_c = run_vm_model(model, device, [m_a, m_b])

    t_at = torch.transpose(t_a, 1, 2) if transpose_a else t_a
    t_bt = torch.transpose(t_b, 1, 2) if transpose_b else t_b
    t_c = torch.matmul(t_at, t_bt)  # pylint: disable=no-member
    check(m_c, t_c, rtol=1e-4, atol=1e-4)
    check(v_c, t_c, rtol=1e-4, atol=1e-4)

    # backward
    m_dc, t_dc = randn_torch(m_c.shape, device=device, dtype=dtype)
    m_c.backward(m_dc)
    t_c.backward(t_dc)

    # Not sure why there is a mismatch. It may be due to the floating errors
    # when collapsing broadcast dimension, but we need more investigations.
    grad_a_tol = 5e-3 if dtype == "float16" and broadcast == "a" else 1e-5
    check(m_a.grad, t_a.grad, rtol=grad_a_tol, atol=grad_a_tol)

    grad_b_tol = 5e-3 if dtype == "float16" and broadcast == "b" else 1e-5
    check(m_b.grad, t_b.grad, rtol=grad_b_tol, atol=grad_b_tol)


if __name__ == "__main__":
    pytest.main([__file__])
