# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import pytest
import raf
from raf.model import Conv2d
from raf.testing import run_infer_type, randn, with_dialect
import tvm
from tvm import relay


def optimize(mod, device="cuda"):
    with raf.device(device):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.DispatchDialect()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
    return mod


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_simple():
    konst, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = konst

        @raf.model.trace
        def forward(self, x):
            y = raf.add(x, self.c)
            y = raf.relu(y)
            y = raf.log(y)
            return y

    def expected(shape):
        add_op = raf._ffi.op.GetOp("raf.op.tvm.add")
        relu_op = raf._ffi.op.GetOp("raf.op.cudnn.relu")
        log_op = raf._ffi.op.GetOp("raf.op.tvm.log")
        null = raf.ir.const(None)

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


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = rand
            self.conv = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @raf.model.trace
        def forward(self, x):
            y = self.conv(x)
            y = raf.add(y, self.c)
            return y

    def expected():
        vec_one = raf.ir.const([1])
        one = raf.ir.const(1)
        nchw = raf.ir.const("NCHW")
        oihw = raf.ir.const("OIHW")
        null = raf.ir.const(None)
        add_op = raf._ffi.op.GetOp("raf.op.tvm.add")
        conv2d_op = raf._ffi.op.GetOp("raf.op.cudnn.conv2d")

        x = raf.ir.var("x", shape=(1, 16, 64, 64))
        c = raf.ir.var("c", shape=(1,))
        w = raf.ir.var("conv.w", shape=(16, 16, 3, 3))
        y = relay.Call(conv2d_op, [x, w, vec_one, vec_one, vec_one, one, nchw, oihw, nchw])
        y = relay.Call(add_op, [y, c, null, null])
        return relay.Function([x, c, w], y)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@with_dialect(["tvm", "cudnn"])
def test_dialect_pref():
    rand, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = rand
            self.conv = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @raf.model.trace
        def forward(self, x):
            y = self.conv(x)
            y = raf.add(y, self.c)
            return y

    def expected():
        vec_one = raf.ir.const([1])
        one = raf.ir.const(1)
        nchw = raf.ir.const("NCHW")
        oihw = raf.ir.const("OIHW")
        null = raf.ir.const(None)
        add_op = raf._ffi.op.GetOp("raf.op.tvm.add")
        conv2d_op = raf._ffi.op.GetOp("raf.op.tvm.conv2d")

        x = raf.ir.var("x", shape=(1, 16, 64, 64))
        c = raf.ir.var("c", shape=(1,))
        w = raf.ir.var("conv.w", shape=(16, 16, 3, 3))
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
