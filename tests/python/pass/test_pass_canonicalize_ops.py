# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, attribute-defined-outside-init, no-self-use
import pytest
import raf
from raf.testing import run_infer_type, randn
from raf.ir import ScopeBuilder
import tvm
from tvm import relay


def test_canonicalize_ops_bias_add_ir():
    x, _ = randn((3, 2, 2))
    bias = raf.array([1, 2], dtype="float32")

    class ModelWithBiasAdd(raf.Model):
        def build(self, bias):
            self.bias = bias

        @raf.model.trace
        def forward(self, x):
            return raf.bias_add(x, self.bias)

    def expected():
        x_var = tvm.relay.var("x", tvm.relay.TensorType(x.shape))
        bias_var = tvm.relay.var("bias", tvm.relay.TensorType(bias.shape, dtype="float32"))
        expand_dim = raf.ir.op.expand_dims(bias_var, 1, 1)
        var_tmp = tvm.relay.var("exp_bias_tmp")
        add = raf.ir.op.add(x_var, var_tmp)
        body = tvm.relay.var("a1")
        body = tvm.relay.Let(body, add, body)
        body = tvm.relay.Let(var_tmp, expand_dim, body)
        return tvm.relay.Function([x_var, bias_var], body)

    model_before = ModelWithBiasAdd(bias)
    # infer type
    mod = raf._ffi.pass_.InferType()(model_before._internal(x).mod)
    # canonicalize ops
    mod = raf._ffi.pass_.CanonicalizeOps()(mod)
    func_canonicalized = raf._ffi.pass_.InferType()(mod)["main"]
    # expected
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(func_canonicalized, func_expected)


def test_canonicalize_ops_multi_bias_add_ir():
    x, _ = randn((3, 2, 2))
    bias = raf.array([1, 2], dtype="float32")

    class ModelWithBiasAdd(raf.Model):
        def build(self, bias):
            self.bias = bias

        @raf.model.trace
        def forward(self, x):
            return raf.bias_add(raf.bias_add(x, self.bias), self.bias)

    def expected(x, bias):
        null = raf.ir.const(None)
        x = relay.var("x", relay.TensorType(x.shape))
        bias = relay.var("bias", relay.TensorType(bias.shape, dtype="float32"))
        sb = ScopeBuilder()
        x_1 = sb.let("x_1", raf.ir.op.expand_dims(bias, raf.ir.const(1), raf.ir.const(1)))
        a_1 = sb.let("a_1", raf.ir.op.add(x, x_1, null, null))
        x_2 = sb.let("x_2", raf.ir.op.expand_dims(bias, raf.ir.const(1), raf.ir.const(1)))
        a_2 = sb.let("a_2", raf.ir.op.add(a_1, x_2, null, null))
        sb.ret(a_2)
        return relay.Function([x, bias], sb.get())

    model_before = ModelWithBiasAdd(bias)
    # infer type
    mod = raf._ffi.pass_.InferType()(model_before._internal(x).mod)
    # canonicalize ops
    mod = raf._ffi.pass_.CanonicalizeOps()(mod)
    func_canonicalized = raf._ffi.pass_.InferType()(mod)["main"]
    # expected
    func_expected = run_infer_type(expected(x, bias))
    assert tvm.ir.structural_equal(func_canonicalized, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
