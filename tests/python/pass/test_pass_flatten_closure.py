# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tvm
from raf.ir import ScopeBuilder
from raf._ffi.pass_ import LambdaLift, FromRelay, FlattenClosure, LiftBranchBody
from tvm import relay


def test_closure():
    def get_mod():
        mod = tvm.IRModule()

        x = relay.var("x", shape=(1, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")  # captured vars
        x_tanh = relay.tanh(x)
        y_tanh = relay.tanh(y)
        add = relay.add(x_tanh, y_tanh)
        closure_vars = [x]
        func = relay.Function(closure_vars, add)

        closure = relay.var("closure")
        let = relay.Let(closure, func, closure)

        z = relay.var("z", shape=(1, 100), dtype="float32")
        body = relay.Call(let, [z])
        mod["main"] = relay.Function([y, z], body)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    assert len(mod.get_global_vars()) == 1
    mod = LambdaLift()(mod)
    mod = FlattenClosure()(mod)
    assert len(mod.get_global_vars()) == 2
    # Check that function body are not closures
    for gvar in mod.get_global_vars():
        assert isinstance(mod[gvar].body, tvm.relay.Let)


def test_while_loop():
    """
    free_var %y
    let %loop =
        fn(%counter, x) {
            if (%counter == 5) {
                return (%counter, %x)
            } else {
                %counter = %counter + 1;
                %x_tanh = relay.tanh(%x)
                %captured = relay.tanh(%y)
                %out = relay.add(%x_tanh, %captured)
                return loop(%counter, %out)
            }
        }; in
        %loop

    return %loop(0, %y)
    """

    def get_recursive_mod():
        sb = ScopeBuilder()
        mod = tvm.IRModule()

        loop = relay.var("loop")
        y = relay.var("y", shape=(1, 100), dtype="float32")
        # Recursive function f
        ti32 = relay.scalar_type("int32")
        counter_var = relay.var("counter", ti32)
        x = relay.var("x", shape=(1, 100), dtype="float32")
        with sb.if_scope(relay.equal(counter_var, relay.const(5, ti32))):
            sb.ret(relay.Tuple([counter_var, x]))
        with sb.else_scope():
            counter = relay.add(counter_var, relay.const(1, ti32))
            x_tanh = relay.tanh(x)
            captured_y_tanh = relay.tanh(y)
            out = relay.add(x_tanh, captured_y_tanh)
            sb.ret(loop(counter, out))

        loop_vars = [counter_var, x]
        func = relay.Function(loop_vars, sb.get())
        let = relay.Let(loop, func, loop)

        body = relay.Call(let, [relay.const(0, ti32), y])
        mod["main"] = relay.Function([y], body)
        return mod

    tvm_mod = get_recursive_mod()
    mod = FromRelay()(tvm_mod)
    assert len(mod.get_global_vars()) == 1
    mod = LambdaLift()(mod)
    mod = FlattenClosure()(mod)
    # Check that function body are not closures
    assert len(mod.get_global_vars()) == 2
    for gvar in mod.get_global_vars():
        assert isinstance(mod[gvar].body, tvm.relay.Let)
    mod = LiftBranchBody()(mod)
    # If else will be lifted
    assert len(mod.get_global_vars()) == 4


if __name__ == "__main__":
    pytest.main([__file__])
