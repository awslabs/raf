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

import pytest
import mnm
from mnm.testing import get_testable_devices, randn, check
import tvm


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_fold_const_model(device, shape):
    const, _ = randn(shape, device=device)

    class ModelWithConst(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.add(self.c, self.c)
            return mnm.add(x, y)

    model = ModelWithConst()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, device=device)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 1 * n_dy
    check(m_dx, n_dx)
    check(m_y, mnm.add(mnm.add(const, const), m_x).numpy())


@pytest.mark.parametrize("device", get_testable_devices()[1:])
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_fold_const_ir(device, shape):
    # pylint: disable=protected-access
    const, _ = randn(shape, device=device)

    class ModelWithConst(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = mnm.matmul(self.c, self.c)
            z = mnm.matmul(x, y)
            return mnm.matmul(x, z)

    def expected():
        x = tvm.relay.var("x", tvm.relay.TensorType(shape))
        c = tvm.relay.var("c", tvm.relay.TensorType(shape))
        # we are only interested in the structure
        t_value = mnm._core.value.TensorValue.from_numpy(const.numpy())
        const_var = mnm._ffi.ir._make.Constant(t_value)
        closure2 = mnm.ir.op.matmul(x, const_var)
        var_a2 = tvm.relay.var("a2")
        var_a3 = tvm.relay.var("a3")
        closure3 = mnm.ir.op.matmul(x, var_a2)
        let3 = tvm.relay.Let(var_a3, closure3, var_a3)
        let2 = tvm.relay.Let(var_a2, closure2, let3)
        return tvm.relay.Function([x, c], let2)

    model_before = ModelWithConst()
    model_before.infer_mode()
    m_x, _ = randn(shape, device=device, requires_grad=True)

    func_before = model_before._internal(m_x).mod["main"]

    # bind parameters
    args = [m_x._ndarray__handle, model_before.c._ndarray__handle]
    func_bound = mnm._ffi.pass_.BindParam(func_before, args)

    # fold constant
    mod = mnm._core.module.IRModule.from_expr(func_bound)
    func_folded = mnm._ffi.pass_.FoldConstant()(mod)["main"]

    func_expected = expected()

    assert tvm.ir.structural_equal(func_folded, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
