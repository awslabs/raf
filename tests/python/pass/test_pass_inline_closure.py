# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-statements
# pylint: disable=no-self-use, too-many-locals
import pytest
import mnm
from mnm.testing import randn
from mnm._core.module import IRModule
from mnm._core.ir_ext import extended_var
from mnm._ffi.pass_ import InlineClosure, InferType, AutoDiff, LambdaLift
from mnm._ffi.pass_ import DeadCodeElimination, InlineBackward
from mnm.ir import MNMSequential, ScopeBuilder
from tvm import relay
import tvm


def test_multi_functions():
    # Create a symbolic model and run it
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    # Get a Relay func
    shape = [3, 3]
    model = Add()
    m_x, _ = randn(shape, requires_grad=True)
    m_y, _ = randn(shape, requires_grad=True)
    record = model._internal(m_x, m_y)
    mod = record.mod

    # Run AutoDiff to get nested functions
    # The backward function will be lifted
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = AutoDiff(record.requires_grads)(mod)
    truth = mod

    # Call Lambda lift pass on the Meta module
    lifted_mod = LambdaLift()(mod)
    assert len(lifted_mod.functions) == 2

    # Invoke the backward closure in main
    fwd, bwd, fwd_var, bwd_var = None, None, None, None
    for k, v in lifted_mod.functions.items():
        if k.name_hint == "main":
            fwd_var, fwd = relay.GlobalVar("fwd"), v
        else:
            bwd_var, bwd = k, v
    x = extended_var("x", shape=(3, 3), dtype="float32")
    y = extended_var("x", shape=(3, 3), dtype="float32")
    dy = extended_var("dy", shape=(3, 3), dtype="float32")
    sb = ScopeBuilder()
    v = sb.let("v", fwd_var)
    v1 = sb.let("v", relay.Call(v, [x, y]))
    v2 = sb.let("v", relay.TupleGetItem(v1, 0))
    v3 = sb.let("v", relay.TupleGetItem(v1, 1))
    v4 = sb.let("v", relay.Call(v3, [dy]))
    v5 = sb.let("v", relay.TupleGetItem(v4, 0))
    v6 = sb.let("v", relay.TupleGetItem(v4, 1))
    v7 = sb.let("v", relay.Tuple([v5, v6]))
    v8 = sb.let("v", relay.Tuple([v2, v7]))
    sb.ret(v8)
    func = relay.Function([x, y, dy], sb.get())
    mod = IRModule({fwd_var: fwd, bwd_var: bwd, relay.GlobalVar("main"): func})
    passes = [InferType(), InlineClosure(), DeadCodeElimination(), InferType()]
    seq = MNMSequential(passes)
    mod = seq(mod)
    record = model._internal(m_x, m_y)
    truth = record.mod
    passes = [InferType(), AutoDiff(record.requires_grads),
              InferType(), DeadCodeElimination(), InlineBackward(), InferType()]
    seq = MNMSequential(passes)
    truth = seq(truth)
    assert tvm.ir.structural_equal(mod['main'], truth['main'])


if __name__ == "__main__":
    pytest.main([__file__])
