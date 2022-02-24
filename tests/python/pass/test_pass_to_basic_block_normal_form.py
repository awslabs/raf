# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import raf
from raf.testing import run_infer_type, randn
import tvm
from tvm import relay


def test_no_bind_tuple():
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
    mod_w_gnf = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod_w_bbnf = raf._ffi.pass_.ToBasicBlockNormalForm()(mod_w_gnf)
    func_after = run_infer_type(mod_w_bbnf)["main"]
    func_expected = run_infer_type(mod_w_gnf)["main"]
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_no_bind_diamond():
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
    mod_w_gnf = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod_w_bbnf = raf._ffi.pass_.ToBasicBlockNormalForm()(mod_w_gnf)
    func_after = run_infer_type(raf._ffi.pass_.ToGraphNormalForm()(mod_w_bbnf))["main"]
    func_expected = run_infer_type(mod_w_gnf)["main"]
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_if():
    def if_expr(x):
        """
        free_var %x: int32
        %0 = equal(%x, 2);
        if (%0) {
          %1 = add(%x, 1);
          multiply(%1, 2)
        } else {
          multiply(%1, 1)
        }
        """
        one = raf.ir.const(1, dtype="int32")
        two = raf.ir.const(2, dtype="int32")
        v1 = raf.ir.op.add(x, one)
        v2 = raf.ir.op.equal(x, two)
        true_branch = raf.ir.op.multiply(v1, two)
        false_branch = raf.ir.op.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        return relay.Function([x], body)

    def expected(x):
        """
        free_var %x: int32
        let %v1: float32 = add(%x, 1);
        %0 = equal(%x, 2);
        if (%0) {
          multiply(%v1, 2)
        } else {
          multiply(%v1, 1)
        }
        """
        one = raf.ir.const(1, dtype="int32")
        two = raf.ir.const(2, dtype="int32")
        v1 = raf.ir.var("v1")
        v2 = raf.ir.op.equal(x, two)
        true_branch = raf.ir.op.multiply(v1, two)
        false_branch = raf.ir.op.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        body = relay.Let(v1, raf.ir.op.add(x, one), body)
        return relay.Function([x], body)

    x = raf.ir.var("x", shape=(), dtype="int32")
    func = if_expr(x)
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected(x))
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_top_level_nested_if():
    cond_t = raf.ir.const(True)
    cond_f = raf.ir.const(False)
    one = raf.ir.const(1.0, dtype="float32")
    three = raf.ir.const(3.0, dtype="float32")

    def nested_if():
        """
        free_var %x: bool
        if (%x) {
          if (True) {
            free_var %z: float32
            %0 = add(%z, %z);
            free_var %y: float32
            %1 = add(%y, %y);
            add(%0, %1)
          } else {
            add(3f, %1)
          }
        } else {
          if (False) {
            %0
          } else {
            1f
          }
        }
        """
        x = raf.ir.var("x", shape=(), dtype="bool")
        y = raf.ir.var("y", shape=(), dtype="float32")
        z = raf.ir.var("z", shape=(), dtype="float32")
        y2 = raf.ir.op.add(y, y)
        z2 = raf.ir.op.add(z, z)
        true_branch = relay.If(cond_t, raf.ir.op.add(z2, y2), raf.ir.op.add(three, y2))
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y, z], body)

    def expected():
        """
        free_var %z: float32
        let %x: float32 = add(%z, %z) /* ty=float32 */;
        free_var %x1: bool
        if (%x1) {
          free_var %y: float32
          let %x2: float32 = add(%y, %y) /* ty=float32 */;
          if (True /* ty=bool */) {
            add(%x, %x2) /* ty=float32 */
          } else {
            add(3f /* ty=float32 */, %x2) /* ty=float32 */
          }
        } else {
          if (False /* ty=bool */) {
            %x
          } else {
            1f /* ty=float32 */
          }
        }
        """
        x = raf.ir.var("x", shape=(), dtype="bool")
        y = raf.ir.var("y", shape=(), dtype="float32")
        z = raf.ir.var("z", shape=(), dtype="float32")
        y2 = relay.var("y2")
        z2 = relay.var("z2")
        true_branch = relay.If(cond_t, raf.ir.op.add(z2, y2), raf.ir.op.add(three, y2))
        true_branch = relay.Let(y2, raf.ir.op.add(y, y), true_branch)
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        body = relay.Let(z2, raf.ir.op.add(z, z), body)
        return relay.Function([x, y, z], body)

    func = nested_if()
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_nested_if():
    cond_t = raf.ir.const(True)
    cond_f = raf.ir.const(False)
    one = raf.ir.const(1.0, dtype="float32")
    two = raf.ir.const(2.0, dtype="float32")
    three = raf.ir.const(3.0, dtype="float32")

    def nested_if():
        """
        free_var %x: bool
        if (%x) {
          if (True) {
            free_var %y: float32
            %0 = add(%y, %y);
            %0
          } else {
            add(3f, %0)
          }
        } else {
          if (False) {
            2f
          } else {
            1f
          }
        }
        """
        x = raf.ir.var("x", shape=(), dtype="bool")
        y = raf.ir.var("y", shape=(), dtype="float32")
        y2 = raf.ir.op.add(y, y)
        true_branch = relay.If(cond_t, y2, raf.ir.op.add(three, y2))
        false_branch = relay.If(cond_f, two, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y], body)

    def expected():
        """
        free_var %x: bool
        if (%x) {
          free_var %y: float32
          let %x1: float32 = add(%y, %y) /* ty=float32 */;
          if (True /* ty=bool */) {
            %x1
          } else {
            add(3f /* ty=float32 */, %x1) /* ty=float32 */
          }
        } else {
          if (False /* ty=bool */) {
            2f /* ty=float32 */
          } else {
            1f /* ty=float32 */
          }
        }
        """
        x = raf.ir.var("x", shape=(), dtype="bool")
        y = raf.ir.var("y", shape=(), dtype="float32")
        y2 = relay.var("y2")
        true_branch = relay.If(cond_t, y2, raf.ir.op.add(three, y2))
        true_branch = relay.Let(y2, raf.ir.op.add(y, y), true_branch)
        false_branch = relay.If(cond_f, two, one)
        body = relay.If(x, true_branch, false_branch)
        return relay.Function([x, y], body)

    func = nested_if()
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)
    relay.analysis.check_basic_block_normal_form(func_after)


def test_may_share():
    shape = (10, 10)
    null = raf.ir.const(None)

    def before():
        in0 = raf.ir.var("in0", shape=shape)
        in1 = raf.ir.var("in1", shape=shape)
        a_1 = raf.ir.op.add(in0, in1, null, null)
        v_0 = raf.ir.var("a2", may_share=in0)
        a_2 = relay.Let(v_0, raf.ir.op.relu(a_1), v_0)
        func = relay.Function([in0, in1], a_2)
        return func

    func = before()
    mod = raf.ir.IRModule()
    mod["main"] = func
    mod_after = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    assert tvm.ir.structural_equal(func_after, run_infer_type(mod)["main"])
    relay.analysis.check_basic_block_normal_form(func_after)


if __name__ == "__main__":
    pytest.main([__file__])
