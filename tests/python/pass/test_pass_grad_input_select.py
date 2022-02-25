# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import raf
from raf._ffi.pass_ import AutoDiff, GradientInputSelection, InferType
from raf._lib import tvm
from raf._lib import relay
from raf.testing import randn, run_infer_type
from raf._core.module import IRModule


def test_conv2d():
    class Model(raf.Model):
        def build(self):
            self.w, _ = randn((1, 1, 3, 3))
            self.w.requires_grad = True

        @raf.model.trace
        def forward(self, x):
            y = raf.conv2d(x, self.w)
            z = raf.relu(y)
            return z

    def expected():
        x = relay.var("x", shape=(1, 1, 224, 224))
        w = relay.var("w", shape=(1, 1, 3, 3))
        dy = relay.var("dy", shape=(1, 1, 222, 222))

        # backward pass closure
        x1 = relay.var("x1")
        x2 = relay.var("x2")
        x3 = relay.var("x3")
        x4 = relay.var("x4")

        closure = relay.var("closure")
        ret = relay.var("ret")
        v = relay.var("a1")
        v1 = relay.var("a2")

        let4 = relay.Let(x4, relay.Tuple((x2, x3)), x4)
        let3 = relay.Let(x3, raf.ir.op.conv2d_dw(x, None, x1, (1, 1, 3, 3), 1, 0, 1, 1), let4)
        let2 = relay.Let(x2, raf.ir.op.conv2d_dx(w, None, x1, (1, 1, 224, 224), 1, 0, 1, 1), let3)
        let1 = relay.Let(x1, raf.ir.op.relu_dx(None, v1, dy), let2)

        let_ret = relay.Let(ret, relay.Tuple((v1, closure)), ret)
        let_closure = relay.Let(closure, relay.Function([dy], let1), let_ret)

        # forward pass
        letv1 = relay.Let(v1, raf.ir.op.relu(v), let_closure)
        letv = relay.Let(v, raf.ir.op.conv2d(x, w), letv1)

        f = relay.Function([x, w], letv)
        return f

    model = Model()
    m_x, _ = randn((1, 1, 224, 224))
    m_x.requires_grad = True
    record = model._internal(m_x)
    mod_before = record.mod
    mod_before = InferType()(mod_before)
    mod_before = AutoDiff(record.requires_grads)(mod_before)
    mod_before = InferType()(mod_before)
    mod = GradientInputSelection()(mod_before)
    func_after = InferType()(mod)["main"]
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_multi_func():
    def multi_func_mod():
        f1 = relay.GlobalVar("f1")  # pylint: disable=invalid-name
        a1 = relay.var("a1")  # pylint: disable=invalid-name
        x = relay.var("x", shape=(1, 100))
        let = relay.Let(a1, raf.ir.op.tanh(x), a1)
        f1_out = relay.Function([x], let)
        mod = IRModule({f1: f1_out})

        a1 = relay.var("a1")  # pylint: disable=invalid-name
        y = relay.var("y", shape=(1, 100))
        let = relay.Let(a1, relay.Call(f1, [y]), a1)
        main_out = relay.Function([y], let)
        mod[relay.GlobalVar("main")] = main_out
        return mod

    mod = multi_func_mod()
    opt_mod = GradientInputSelection()(mod)
    assert tvm.ir.structural_equal(mod["f1"], opt_mod["f1"])
    assert tvm.ir.structural_equal(mod["main"], opt_mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
