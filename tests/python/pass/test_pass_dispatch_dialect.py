# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import pytest
import mnm
from mnm.model import Conv2d
from mnm.testing import run_infer_type, randn, with_dialect
import tvm
from tvm import relay


def optimize(mod, device="cuda"):
    with mnm.device(device):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.DispatchDialect()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
    return mod


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_simple():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.c)
            y = mnm.relu(y)
            y = mnm.log(y)
            return y

    def expected(shape):
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.cudnn.relu")
        log_op = mnm._ffi.op.GetOp("mnm.op.tvm.log")
        null = mnm.ir.const(None)

        x = relay.var("x", shape=shape)
        y = relay.var("c", shape=(1,))
        z = relay.Call(add_op, [x, y, null, null])
        z = relay.Call(log_op, [relay.Call(relu_op, [z])])
        return relay.Function([x, y], z)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @mnm.model.trace
        def forward(self, x):
            y = self.conv(x)
            y = mnm.add(y, self.c)
            return y

    def expected():
        vec_one = mnm.ir.const([1])
        one = mnm.ir.const(1)
        nchw = mnm.ir.const("NCHW")
        oihw = mnm.ir.const("OIHW")
        null = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.cudnn.conv2d")

        x = mnm.ir.var("x", shape=(1, 16, 64, 64))
        c = mnm.ir.var("c", shape=(1,))
        w = mnm.ir.var("conv.w", shape=(16, 16, 3, 3))
        y = relay.Call(conv2d_op, [x, w, vec_one, vec_one, vec_one, one, nchw, oihw, nchw])
        y = relay.Call(add_op, [y, c, null, null])
        return relay.Function([x, c, w], y)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@with_dialect(["tvm", "cudnn"])
def test_dialect_pref():
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @mnm.model.trace
        def forward(self, x):
            y = self.conv(x)
            y = mnm.add(y, self.c)
            return y

    def expected():
        vec_one = mnm.ir.const([1])
        one = mnm.ir.const(1)
        nchw = mnm.ir.const("NCHW")
        oihw = mnm.ir.const("OIHW")
        null = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.tvm.conv2d")

        x = mnm.ir.var("x", shape=(1, 16, 64, 64))
        c = mnm.ir.var("c", shape=(1,))
        w = mnm.ir.var("conv.w", shape=(16, 16, 3, 3))
        y = relay.Call(conv2d_op, [x, w, vec_one, vec_one, vec_one, one, nchw, oihw, nchw])
        y = relay.Call(add_op, [y, c, null, null])
        return relay.Function([x, c, w], y)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
