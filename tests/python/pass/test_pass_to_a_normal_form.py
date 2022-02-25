# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import raf
from raf.ir import ScopeBuilder
from raf.testing import run_infer_type, randn
from raf.model import Conv2d
import tvm
from tvm import relay


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

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
            self.conv2 = Conv2d(16, 16, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)

        @raf.model.trace
        def forward(self, x):
            x = raf.add(x, self.c)
            y = self.conv1(x)
            # this is the next dominator.
            y1 = raf.add(y, self.c)
            y = raf.add(y, y1)
            # second path
            z2 = self.conv2(y)
            z3 = self.conv3(y)
            # add can only be fused to z1
            z = raf.add(z2, z3)
            return z

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_tuple():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            zz = raf.split(z, 2)
            return zz[0]

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_diamond():
    konst, _ = randn((1,))

    class Model(raf.Model):
        def build(self):
            self.c = konst

        @raf.model.trace
        def forward(self, x, y):
            z1 = raf.add(x, y)
            z2 = raf.multiply(x, self.c)
            return raf.relu(raf.add(z1, z2))

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_may_share():
    shape = (10, 10)
    null = raf.ir.const(None)

    def before():
        in0 = raf.ir.var("in0", shape=shape)
        in1 = raf.ir.var("in1", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", raf.ir.op.add(in0, in1, null, null))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1), may_share=in0)
        a_3 = sb.let("a3", raf.ir.op.relu(a_2))
        sb.ret(a_3)
        func = relay.Function([in0, in1], sb.get())
        return func

    func = before()
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod_after = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    assert tvm.ir.structural_equal(func_after, run_infer_type(before()))


def test_if():
    def before():
        """
        free_var %data: int32
        let %v1: float32 = add(%data, 1);
        %0 = equal(%x, 2);
        if (%0) {
          multiply(%v1, 2)
        } else {
          multiply(%v1, 1)
        }
        """
        data = raf.ir.var("data", shape=(), dtype="int32")
        one = raf.ir.const(1, dtype="int32")
        two = raf.ir.const(2, dtype="int32")
        v1 = raf.ir.var("v1")
        v2 = raf.ir.op.equal(data, two)
        true_branch = raf.ir.op.multiply(v1, two)
        false_branch = raf.ir.op.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        body = relay.Let(v1, raf.ir.op.add(data, one), body)
        return relay.Function([data], body)

    def expected():
        """
        free_var %data: int32
        let %v1 = add(%data, 1);
        let %x3 = equal(%data, 2);
        let %x4 = if (%x3) {
          let %x2 = multiply(%v1, 2);
          %x2
        } else {
          let %x1 = multiply(%v1, 1);
          %x1
        }
        %x4
        """
        data = raf.ir.var("data", shape=(), dtype="int32")
        one = raf.ir.const(1, dtype="int32")
        two = raf.ir.const(2, dtype="int32")

        sb = ScopeBuilder()
        v_1 = sb.let("v1", raf.ir.op.add(data, one))
        x_3 = sb.let("x3", raf.ir.op.equal(data, two))

        sb_true = ScopeBuilder()
        x_2 = sb_true.let("x2", raf.ir.op.multiply(v_1, two))
        sb_true.ret(x_2)

        sb_false = ScopeBuilder()
        x_1 = sb_false.let("x1", raf.ir.op.multiply(v_1, one))
        sb_false.ret(x_1)

        x_4 = sb.let("x4", relay.If(x_3, sb_true.get(), sb_false.get()))
        sb.ret(x_4)
        return relay.Function([data], sb.get())

    func = before()
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
