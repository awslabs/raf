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

    def expected():
        x = relay.var("x", shape=(10, 20))
        c = relay.var("c", shape=(1,))
        y = mnm.ir.op.add(x, c)
        y = mnm.ir.op.log(mnm.ir.op.relu(y))
        f = relay.Function([x, c], y)
        return f

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_tuple():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            zz = mnm.split(z, 2)
            return zz[0]

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        z = mnm.ir.op.add(x, y)
        z = mnm.ir.op.split(z, 2)
        z = relay.TupleGetItem(z, 0)
        f = relay.Function([x, y], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


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

    def expected():
        x = relay.var("x", shape=(10, 20))
        y = relay.var("y", shape=(10, 1))
        c = relay.var("c", shape=(1,))
        z1 = mnm.ir.op.add(x, y)
        z2 = mnm.ir.op.multiply(x, c)
        z = mnm.ir.op.add(z1, z2)
        z = mnm.ir.op.relu(z)
        f = relay.Function([x, y, c], z)
        return f

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_after = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_may_share():
    shape = (10, 10)
    null = mnm.ir.const(None)

    def before():
        in0 = mnm.ir.var("in0", shape=shape)
        in1 = mnm.ir.var("in1", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", mnm.ir.op.add(in0, in1, null, null))
        a_2 = sb.let("a2", mnm.ir.op.relu(a_1), may_share=in0)  # This let should be preserved.
        sb.ret(a_2)
        func = relay.Function([in0, in1], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        in0 = mnm.ir.var("in0", shape=shape)
        in1 = mnm.ir.var("in1", shape=shape)
        a_1 = mnm.ir.op.add(in0, in1, null, null)
        v_0 = mnm.ir.var("a2", may_share=in0)
        a_2 = relay.Let(v_0, mnm.ir.op.relu(a_1), v_0)
        func = relay.Function([in0, in1], a_2)
        return func

    mod = before()
    after_func = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(mod))["main"]
    expected_func = run_infer_type(expected())
    assert tvm.ir.structural_equal(after_func, expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
