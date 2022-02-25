# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,protected-access,too-many-statements
# pylint: disable=no-self-use, too-many-locals
import pytest
import raf
from raf.testing import randn
from raf._core.module import IRModule
from raf._core.ir_ext import extended_var
from raf._ffi.pass_ import InlineClosure, InferType, AutoDiff, LambdaLift
from raf._ffi.pass_ import DeadCodeElimination, InlineBackward
from raf.ir import RAFSequential, ScopeBuilder
from raf.model.nn import BatchNorm
from tvm import relay
import tvm


def verify_ir(model, args, reconstructor):
    """A helper function to process the module and verify the transformed IR."""
    record = model._internal(*args)
    mod = record.mod

    # Run AutoDiff to get nested functions, and the backward function will be lifted later.
    mod = raf._ffi.pass_.InferType()(mod)
    mod = AutoDiff(record.requires_grads)(mod)

    # Call Lambda lift pass on the RAF module
    lifted_mod = LambdaLift()(mod)

    # Extract forward and backward closures to reconstruct the main function.
    fwd, bwd, fwd_var, bwd_var = None, None, None, None
    assert len(lifted_mod.functions) == 2
    for var, func in lifted_mod.functions.items():
        if var.name_hint == "main":
            fwd_var, fwd = relay.GlobalVar("fwd"), func
        else:
            bwd_var, bwd = var, func

    mod = reconstructor(fwd, bwd, fwd_var, bwd_var)
    seq = RAFSequential([InferType(), InlineClosure(), DeadCodeElimination(), InferType()])
    mod = seq(mod)

    # Make a ground truth.
    record = model._internal(*args)
    truth = record.mod
    seq = RAFSequential(
        [
            InferType(),
            AutoDiff(record.requires_grads),
            InferType(),
            DeadCodeElimination(),
            InlineBackward(),
            InferType(),
        ]
    )
    truth = seq(truth)
    assert tvm.ir.structural_equal(mod["main"], truth["main"]), raf.ir.AsText(mod["main"])


def test_basic():
    class Add(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            return raf.add(x, y)

    # Get a Relay func
    shape = [3, 3]
    model = Add()
    m_x, _ = randn(shape, requires_grad=True)
    m_y, _ = randn(shape, requires_grad=True)

    def _reconstructor(fwd, bwd, fwd_var, bwd_var):
        # Reconstruct the main function that invokes fwd and bwd.
        sb = ScopeBuilder()
        x = extended_var("x", shape=(3, 3), dtype="float32")
        y = extended_var("y", shape=(3, 3), dtype="float32")
        dy = extended_var("dy", shape=(3, 3), dtype="float32")
        f_v = sb.let("v", fwd_var)
        v_1 = sb.let("v1", relay.Call(f_v, [x, y]))
        v_2 = sb.let("v2", relay.TupleGetItem(v_1, 0))
        v_3 = sb.let("v3", relay.TupleGetItem(v_1, 1))
        v_4 = sb.let("v4", relay.Call(v_3, [dy]))
        v_5 = sb.let("v5", relay.TupleGetItem(v_4, 0))
        v_6 = sb.let("v6", relay.TupleGetItem(v_4, 1))
        v_7 = sb.let("v7", relay.Tuple([v_5, v_6]))
        v_8 = sb.let("v8", relay.Tuple([v_2, v_7]))
        sb.ret(v_8)
        func = relay.Function([x, y, dy], sb.get())
        return IRModule({fwd_var: fwd, bwd_var: bwd, relay.GlobalVar("main"): func})

    verify_ir(model, [m_x, m_y], _reconstructor)


def test_inplace():
    """Test a model with batch_norm which has inplace updated parameters."""

    class Model(raf.Model):
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @raf.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = raf.relu(x)
            return x

    # Get a Relay func
    shape = (2, 3, 4, 5)
    model = Model(num_features=shape[1])
    model.train_mode()
    m_x, _ = randn(shape, requires_grad=True)

    def _reconstructor(fwd, bwd, fwd_var, bwd_var):
        # Reconstruct the main function that invokes fwd and bwd.
        sb = ScopeBuilder()
        x = extended_var("x", shape=shape, dtype="float32")
        bn_b = extended_var("bn_b", shape=(shape[1],), dtype="float32")
        bn_m = extended_var("bn_m", shape=(shape[1],), dtype="float32")
        bn_v = extended_var("bn_v", shape=(shape[1],), dtype="float32")
        bn_w = extended_var("bn_w", shape=(shape[1],), dtype="float32")
        dy = extended_var(
            "dy",
            type_annotation=relay.TupleType(
                [
                    relay.TensorType(shape=shape, dtype="float32"),
                    relay.TensorType(shape=(shape[1],), dtype="float32"),
                    relay.TensorType(shape=(shape[1],), dtype="float32"),
                ]
            ),
        )
        f_v = sb.let("v", fwd_var)
        v_1 = sb.let("v1", relay.Call(f_v, [x, bn_b, bn_m, bn_v, bn_w]))
        v_2 = sb.let("v2", relay.TupleGetItem(v_1, 0))  # Forward output[0]: (y, bn_m, bn_v)
        v_3 = sb.let("v3", relay.TupleGetItem(v_1, 1))  # Forward output[1]: backward closure
        v_m = sb.let("vm", relay.TupleGetItem(v_2, 1))  # Forward output[0][1]: bn_m
        v_v = sb.let("vv", relay.TupleGetItem(v_2, 2))  # Forward output[0][2]: bn_v

        dy_0 = sb.let("dy_0", relay.TupleGetItem(dy, 0))
        v_t = sb.let("v_t", relay.Tuple([dy_0, v_m, v_v]))  # Backward intput: (dy.0, bn_m, bn_v)
        v_4 = sb.let("v4", relay.Call(v_3, [v_t]))
        v_5 = sb.let("v5", relay.TupleGetItem(v_4, 0))
        v_6 = sb.let("v6", relay.TupleGetItem(v_4, 1))
        v_7 = sb.let("v7", relay.TupleGetItem(v_4, 2))
        v_8 = sb.let("v8", relay.TupleGetItem(v_4, 3))
        v_9 = sb.let("v9", relay.TupleGetItem(v_4, 4))
        v_10 = sb.let("v10", relay.Tuple([v_5, v_6, v_7, v_8, v_9]))
        v_11 = sb.let("v11", relay.Tuple([v_2, v_10]))  # Output: (forward output[0], gradients)
        sb.ret(v_11)
        func = relay.Function([x, bn_b, bn_m, bn_v, bn_w, dy], sb.get())
        mod = IRModule({fwd_var: fwd, bwd_var: bwd, relay.GlobalVar("main"): func})
        return mod

    verify_ir(model, [m_x], _reconstructor)


def test_no_let():
    """Test the closure with a single return variable."""
    shape = (2, 2)

    sb = ScopeBuilder()
    x = extended_var("x", shape=shape, dtype="float32")
    sb.ret(x)
    func1 = relay.Function([x], sb.get())
    fun_var = relay.GlobalVar("func1")

    sb = ScopeBuilder()
    y = extended_var("y", shape=shape, dtype="float32")
    v_0 = sb.let("v0", relay.Call(fun_var, [y]))
    sb.ret(v_0)
    func2 = relay.Function([y], sb.get())
    mod = IRModule({fun_var: func1, relay.GlobalVar("main"): func2})
    seq = RAFSequential([InferType(), InlineClosure(), DeadCodeElimination(), InferType()])
    mod = seq(mod)

    def expected():
        sb = ScopeBuilder()
        y = extended_var("y", shape=shape, dtype="float32")
        sb.ret(y)
        func = relay.Function([y], sb.get())
        mod = IRModule.from_expr(func)
        mod = InferType()(mod)
        return mod

    assert tvm.ir.structural_equal(mod["main"], expected()["main"]), raf.ir.AsText(mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
