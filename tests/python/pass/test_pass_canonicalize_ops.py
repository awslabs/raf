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

# pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
import pytest
import mnm
from mnm.testing import run_infer_type, randn
from mnm.ir import ScopeBuilder
import tvm
from tvm import relay


def test_canonicalize_ops_bias_add_ir():
    x, _ = randn((3, 2, 2))
    bias = mnm.array([1, 2], dtype="float32")

    class ModelWithBiasAdd(mnm.Model):
        def build(self, bias):
            self.bias = bias

        @mnm.model.trace
        def forward(self, x):
            return mnm.bias_add(x, self.bias)

    def expected():
        x_var = tvm.relay.var("x", tvm.relay.TensorType(x.shape))
        bias_var = tvm.relay.var("bias", tvm.relay.TensorType(bias.shape, dtype="float32"))
        expand_dim = mnm.ir.op.expand_dims(bias_var, 1, 1)
        var_tmp = tvm.relay.var("exp_bias_tmp")
        add = mnm.ir.op.add(x_var, var_tmp)
        body = tvm.relay.var("a1")
        body = tvm.relay.Let(body, add, body)
        body = tvm.relay.Let(var_tmp, expand_dim, body)
        return tvm.relay.Function([x_var, bias_var], body)

    model_before = ModelWithBiasAdd(bias)
    # infer type
    mod = mnm._ffi.pass_.InferType()(model_before._internal(x).mod)
    # canonicalize ops
    mod = mnm._ffi.pass_.CanonicalizeOps()(mod)
    func_canonicalized = mnm._ffi.pass_.InferType()(mod)["main"]
    # expected
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_canonicalized, func_expected)


def test_canonicalize_ops_multi_bias_add_ir():
    x, _ = randn((3, 2, 2))
    bias = mnm.array([1, 2], dtype="float32")

    class ModelWithBiasAdd(mnm.Model):
        def build(self, bias):
            self.bias = bias

        @mnm.model.trace
        def forward(self, x):
            return mnm.bias_add(mnm.bias_add(x, self.bias), self.bias)

    def expected(x, bias):
        null = mnm.ir.const(None)
        x = relay.var("x", relay.TensorType(x.shape))
        bias = relay.var("bias", relay.TensorType(bias.shape, dtype="float32"))
        sb = ScopeBuilder()
        x_1 = sb.let("x_1", mnm.ir.op.expand_dims(bias, mnm.ir.const(1), mnm.ir.const(1)))
        a_1 = sb.let("a_1", mnm.ir.op.add(x, x_1, null, null))
        x_2 = sb.let("x_2", mnm.ir.op.expand_dims(bias, mnm.ir.const(1), mnm.ir.const(1)))
        a_2 = sb.let("a_2", mnm.ir.op.add(a_1, x_2, null, null))
        sb.ret(a_2)
        return relay.Function([x, bias], sb.get())

    model_before = ModelWithBiasAdd(bias)
    # infer type
    mod = mnm._ffi.pass_.InferType()(model_before._internal(x).mod)
    # canonicalize ops
    mod = mnm._ffi.pass_.CanonicalizeOps()(mod)
    func_canonicalized = mnm._ffi.pass_.InferType()(mod)["main"]
    # expected
    func_expected = run_infer_type(expected(x, bias))
    assert tvm.ir.structural_equal(func_canonicalized, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
