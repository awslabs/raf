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

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import mnm
from mnm.ir import ScopeBuilder
from mnm.testing import run_infer_type, randn
from mnm.model import Conv2d
import tvm
from tvm import relay


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

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
            self.conv2 = Conv2d(16, 16, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)

        @mnm.model.trace
        def forward(self, x):
            x = mnm.add(x, self.c)
            y = self.conv1(x)
            # this is the next dominator.
            y1 = mnm.add(y, self.c)
            y = mnm.add(y, y1)
            # second path
            z2 = self.conv2(y)
            z3 = self.conv3(y)
            # add can only be fused to z1
            z = mnm.add(z2, z3)
            return z

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_tuple():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            zz = mnm.split(z, 2)
            return zz[0]

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_diamond():
    konst, _ = randn((1,))

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x, y):
            z1 = mnm.add(x, y)
            z2 = mnm.multiply(x, self.c)
            return mnm.relu(mnm.add(z1, z2))

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_may_share():
    shape = (10, 10)
    null = mnm.ir.const(None)

    def before():
        in0 = mnm.ir.var("in0", shape=shape)
        in1 = mnm.ir.var("in1", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", mnm.ir.op.add(in0, in1, null, null))
        a_2 = sb.let("a2", mnm.ir.op.relu(a_1), may_share=in0)
        a_3 = sb.let("a3", mnm.ir.op.relu(a_2))
        sb.ret(a_3)
        func = relay.Function([in0, in1], sb.get())
        return func

    func = before()
    mod = mnm.ir.IRModule()
    mod["main"] = func
    mod_after = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod_after = mnm._ffi.pass_.ToANormalForm()(mod)
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
        data = mnm.ir.var("data", shape=(), dtype="int32")
        one = mnm.ir.const(1, dtype="int32")
        two = mnm.ir.const(2, dtype="int32")
        v1 = mnm.ir.var("v1")
        v2 = mnm.ir.op.equal(data, two)
        true_branch = mnm.ir.op.multiply(v1, two)
        false_branch = mnm.ir.op.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        body = relay.Let(v1, mnm.ir.op.add(data, one), body)
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
        data = mnm.ir.var("data", shape=(), dtype="int32")
        one = mnm.ir.const(1, dtype="int32")
        two = mnm.ir.const(2, dtype="int32")

        sb = ScopeBuilder()
        v_1 = sb.let("v1", mnm.ir.op.add(data, one))
        x_3 = sb.let("x3", mnm.ir.op.equal(data, two))

        sb_true = ScopeBuilder()
        x_2 = sb_true.let("x2", mnm.ir.op.multiply(v_1, two))
        sb_true.ret(x_2)

        sb_false = ScopeBuilder()
        x_1 = sb_false.let("x1", mnm.ir.op.multiply(v_1, one))
        sb_false.ret(x_1)

        x_4 = sb.let("x4", relay.If(x_3, sb_true.get(), sb_false.get()))
        sb.ret(x_4)
        return relay.Function([data], sb.get())

    func = before()
    mod = mnm.ir.IRModule()
    mod["main"] = func
    mod_after = mnm._ffi.pass_.ToANormalForm()(mod)
    func_after = run_infer_type(mod_after)["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
