# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tvm
from raf._ffi.pass_ import FromRelay, InferType, LiftBranchBody
from raf.ir import ScopeBuilder
from tvm import relay


def test_basic_if():
    main = relay.GlobalVar("main")

    def get_recursive_mod():
        sb = ScopeBuilder()
        mod = tvm.IRModule()

        # Recursive function f
        ti32 = relay.scalar_type("int32")
        n = relay.var("n", ti32)
        x = relay.var("x", shape=(1, 100), dtype="float32")
        with sb.if_scope(relay.equal(n, relay.const(0, ti32))):
            sb.ret(relay.tanh(x))
        with sb.else_scope():
            sb.ret(relay.sigmoid(x))
        mod[main] = relay.Function([n, x], sb.get())
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_recursive_mod()
    mod = FromRelay()(tvm_mod)
    mod = InferType()(mod)
    mod = LiftBranchBody()(mod)
    mod = InferType()(mod)
    assert len(mod.get_global_vars()) == 3


def test_raf_recursive_function():
    f1 = relay.GlobalVar("f1")  # pylint: disable=invalid-name
    main = relay.GlobalVar("main")

    def get_recursive_mod():
        sb = ScopeBuilder()
        mod = tvm.IRModule()

        # Recursive function f
        ti32 = relay.scalar_type("int32")
        n = relay.var("n", ti32)
        x = relay.var("x", shape=(1, 100), dtype="float32")
        with sb.if_scope(relay.equal(n, relay.const(0, ti32))):
            sb.ret(x)
        with sb.else_scope():
            sb.ret(f1(relay.subtract(n, relay.const(1, ti32)), relay.tanh(x)))
        mod[f1] = relay.Function([n, x], sb.get())
        mod = relay.transform.InferType()(mod)

        n1 = relay.var("n1", ti32)  # pylint: disable=invalid-name
        y = relay.var("y", shape=(1, 100), dtype="float32")
        out = f1(n1, y)
        mod[main] = relay.Function([n1, y], out)
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_recursive_mod()
    mod = FromRelay()(tvm_mod)
    mod = InferType()(mod)
    mod = LiftBranchBody()(mod)
    mod = InferType()(mod)

    # Check that the true branch is a Let expression
    for gvar in mod.get_global_vars():
        if "true" in gvar.name_hint:
            assert isinstance(mod[gvar].body, relay.Let)

    t_0 = relay.scalar_type(dtype="int32")
    t_1 = relay.TensorType((1, 100))
    t_2 = relay.TensorType((1, 100))
    expected_ty = relay.FuncType([t_0, t_1], t_2)
    assert mod["f1"].checked_type == expected_ty
    assert mod["main"].checked_type == expected_ty
    assert len(mod.get_global_vars()) == 4


if __name__ == "__main__":
    pytest.main([__file__])
