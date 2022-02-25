# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-statements
# pylint: disable=no-self-use
import pytest
import raf
from raf._core.ndarray import Symbol
from raf.testing import randn, run_infer_type
import tvm
from tvm import relay


def test_inline():
    class Model(raf.Model):
        def build(self):
            self.w, _ = randn((20, 15), device="cpu")

        @raf.model.trace
        def forward(self, x, dy):
            xt = raf.transpose(x)
            dw = raf.matmul(xt, dy)
            tup = Symbol.make_tuple(
                [
                    dw._Symbol__handle,
                ]
            )
            it = tup[0]
            y = raf.subtract(self.w, it)
            return y

    def expected1():
        x = raf.ir.var("x", shape=(10, 20))
        dy = raf.ir.var("dy", shape=(10, 15))
        w = raf.ir.var("dy", shape=(20, 15))
        a1 = raf.ir.var("a1")
        a2 = raf.ir.var("a2")
        a3 = raf.ir.var("a3")
        a5 = raf.ir.var("a5")
        let5 = relay.Let(a5, raf.ir.op.subtract(w, a2), a5)
        let3 = relay.Let(a3, relay.Tuple([a2]), let5)
        let2 = relay.Let(a2, raf.ir.op.matmul(a1, dy), let3)
        let1 = relay.Let(a1, raf.ir.op.transpose(x), let2)
        return relay.Function([x, dy, w], let1)

    def expected2():
        x = raf.ir.var("x", shape=(10, 20))
        dy = raf.ir.var("dy", shape=(10, 15))
        w = raf.ir.var("dy", shape=(20, 15))
        a1 = raf.ir.var("a1")
        a2 = raf.ir.var("a2")
        a5 = raf.ir.var("a5")
        let5 = relay.Let(a5, raf.ir.op.subtract(w, a2), a5)
        let2 = relay.Let(a2, raf.ir.op.matmul(a1, dy), let5)
        let1 = relay.Let(a1, raf.ir.op.transpose(x), let2)
        return relay.Function([x, dy, w], let1)

    def expected3():
        x = raf.ir.var("p0", shape=(20, 15))
        y = raf.ir.var("p1", shape=(20, 10))
        z = raf.ir.var("p2", shape=(10, 15))
        p3 = raf.ir.var("p", relay.TupleType(()))
        p4 = raf.ir.var("p", relay.TupleType(()))
        out = relay.Call(raf._ffi.op.GetOp("raf.op.tvm.matmul"), [y, z])
        out = relay.Call(raf._ffi.op.GetOp("raf.op.tvm.subtract"), [x, out, p3, p4])
        f = relay.Function([x, y, z, p3, p4], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "tvm")

        x = raf.ir.var("x", shape=(10, 20))
        dy = raf.ir.var("dy", shape=(10, 15))
        w = raf.ir.var("dy", shape=(20, 15))
        null = raf.ir.const(None)
        out = raf.ir.op.transpose(x)
        out = relay.Call(f, [w, out, dy, null, null])
        return relay.Function([x, dy, w], out)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    m_dy, _ = randn((10, 15), device="cpu")
    mod = model._internal(m_x, m_dy).mod

    mod = run_infer_type(mod)
    mod = run_infer_type(raf._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected1())
    assert tvm.ir.structural_equal(mod["main"], func_expected)

    mod = run_infer_type(raf._ffi.pass_.DeadCodeElimination()(mod))
    func_expected = run_infer_type(expected2())
    assert tvm.ir.structural_equal(mod["main"], func_expected)

    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.InferType()(mod)
    mod = raf._ffi.pass_.FuseTVM()(mod)
    func_expected = run_infer_type(expected3())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


def test_nested_tuple():
    shape = (10, 20)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            tup1 = Symbol.make_tuple(
                [
                    x,
                ]
            )
            tup2 = Symbol.make_tuple(
                [
                    y,
                ]
            )
            tup = Symbol.make_tuple([tup1, tup2])
            x1 = tup[0]
            y1 = x1[0]
            return y1

    def expected():
        x = raf.ir.var("x", shape=shape)
        y = raf.ir.var("y", shape=shape)
        a1 = raf.ir.var("a1")
        a2 = raf.ir.var("a2")
        a3 = raf.ir.var("a3")
        let3 = relay.Let(a3, relay.Tuple([a1, a2]), x)
        let2 = relay.Let(
            a2,
            relay.Tuple(
                [
                    y,
                ]
            ),
            let3,
        )
        let1 = relay.Let(
            a1,
            relay.Tuple(
                [
                    x,
                ]
            ),
            let2,
        )
        return relay.Function([x, y], let1)

    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    model = Model()
    mod = model._internal(m_x, m_y).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(raf._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


def test_tuple_sequence():
    shape = (10, 20)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            a = Symbol.make_tuple(
                [
                    x,
                ]
            )
            a = Symbol.make_tuple(
                [
                    a[0],
                ]
            )
            a = raf.add(a[0], y)
            return a

    def expected():
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        let3 = relay.Let(a3, raf.ir.op.add(x, y), a3)
        let2 = relay.Let(
            a2,
            relay.Tuple(
                [
                    x,
                ]
            ),
            let3,
        )
        let1 = relay.Let(
            a1,
            relay.Tuple(
                [
                    x,
                ]
            ),
            let2,
        )
        return relay.Function([x, y], let1)

    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    model = Model()
    mod = model._internal(m_x, m_y).mod
    mod = run_infer_type(mod)
    mod = run_infer_type(raf._ffi.pass_.InlineLet()(mod))
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
