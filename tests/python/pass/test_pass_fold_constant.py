# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import raf
from raf.testing import get_testable_devices, randn, check
import tvm


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_fold_const_model(device, shape):
    const, _ = randn(shape, device=device)

    class ModelWithConst(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = raf.add(self.c, self.c)
            return raf.add(x, y)

    model = ModelWithConst()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, device=device)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 1 * n_dy
    check(m_dx, n_dx)
    check(m_y, raf.add(raf.add(const, const), m_x).numpy())


@pytest.mark.parametrize("device", get_testable_devices()[1:])
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_fold_const_ir(device, shape):
    # pylint: disable=protected-access
    const, _ = randn(shape, device=device)

    class ModelWithConst(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.c = const

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            y = raf.matmul(self.c, self.c)
            z = raf.matmul(x, y)
            return raf.matmul(x, z)

    def expected():
        x = tvm.relay.var("x", tvm.relay.TensorType(shape))
        c = tvm.relay.var("c", tvm.relay.TensorType(shape))
        # we are only interested in the structure
        t_value = raf._core.value.TensorValue.from_numpy(const.numpy())
        const_var = raf._ffi.ir._make.Constant(t_value)
        closure2 = raf.ir.op.matmul(x, const_var)
        var_a2 = tvm.relay.var("a2")
        var_a3 = tvm.relay.var("a3")
        closure3 = raf.ir.op.matmul(x, var_a2)
        let3 = tvm.relay.Let(var_a3, closure3, var_a3)
        let2 = tvm.relay.Let(var_a2, closure2, let3)
        return tvm.relay.Function([x, c], let2)

    model_before = ModelWithConst()
    model_before.infer_mode()
    m_x, _ = randn(shape, device=device, requires_grad=True)

    func_before = model_before._internal(m_x).mod["main"]

    # bind parameters
    args = [m_x._ndarray__handle, model_before.c._ndarray__handle]
    func_bound = raf._ffi.pass_.BindParam(func_before, args)

    # fold constant
    mod = raf._core.module.IRModule.from_expr(func_bound)
    func_folded = raf._ffi.pass_.FoldConstant()(mod)["main"]

    func_expected = expected()

    assert tvm.ir.structural_equal(func_folded, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
