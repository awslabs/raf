# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import raf
from raf.ir import ScopeBuilder
from raf.testing import run_infer_type, randn
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

    def expected():
        x = relay.var("x", shape=(10, 20))
        c = relay.var("c", shape=(1,))
        y = raf.ir.op.add(x, c)
        y = raf.ir.op.log(raf.ir.op.relu(y))
        f = relay.Function([x, c], y)
        return f

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_after = run_infer_type(raf._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_tuple():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = raf.add(x, y)
            zz = raf.split(z, 2)
            return zz[0]

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        z = raf.ir.op.add(x, y)
        z = raf.ir.op.split(z, 2)
        z = relay.TupleGetItem(z, 0)
        f = relay.Function([x, y], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(raf._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


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

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        c = relay.var("c", shape=(1,))
        z1 = raf.ir.op.add(x, y)
        z2 = raf.ir.op.multiply(x, c)
        z = raf.ir.op.add(z1, z2)
        z = raf.ir.op.relu(z)
        f = relay.Function([x, y, c], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(raf._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_may_share():
    shape = (10, 10)
    null = raf.ir.const(None)

    def before():
        in0 = raf.ir.var("in0", shape=shape)
        in1 = raf.ir.var("in1", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", raf.ir.op.add(in0, in1, null, null))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1), may_share=in0)  # This let should be preserved.
        sb.ret(a_2)
        func = relay.Function([in0, in1], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        in0 = raf.ir.var("in0", shape=shape)
        in1 = raf.ir.var("in1", shape=shape)
        a_1 = raf.ir.op.add(in0, in1, null, null)
        v_0 = raf.ir.var("a2", may_share=in0)
        a_2 = relay.Let(v_0, raf.ir.op.relu(a_1), v_0)
        func = relay.Function([in0, in1], a_2)
        return func

    mod = before()
    after_func = run_infer_type(raf._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    expected_func = run_infer_type(expected())
    assert tvm.ir.structural_equal(after_func, expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
