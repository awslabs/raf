# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tvm
import raf
from raf.testing import randn, get_testable_devices
from raf._ffi.pass_ import AutoDiff, LambdaLift, FromRelay, LiftBranchBody, InferType
from tvm import relay


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_basic(device, shape):
    # pylint: disable=protected-access
    # Create a symbolic model and run it
    class Add(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):  # pylint: disable=no-self-use
            return raf.add(x, y)

    # Get a Relay func
    model = Add()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y, _ = randn(shape, device=device, requires_grad=True)
    record = model._internal(m_x, m_y)
    mod = record.mod

    # Run AutoDiff to get nested functions
    # The backward function will be lifted
    mod = AutoDiff(record.requires_grads)(InferType()(mod))

    # Call Lambda lift pass on the RAF module
    lifted_mod = LambdaLift()(mod)

    assert len(lifted_mod.functions) == 2


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
        sb = raf.ir.ScopeBuilder()
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
    try:
        mod = LiftBranchBody()(mod)
        assert False, "LiftBranchBody pass should have failed"
    except:  # pylint: disable=bare-except
        mod = LambdaLift()(mod)
        mod = LiftBranchBody()(mod)
    assert len(mod.get_global_vars()) == 4


if __name__ == "__main__":
    pytest.main([__file__])
