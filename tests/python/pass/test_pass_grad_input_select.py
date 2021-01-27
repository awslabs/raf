# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest
import mnm
from mnm._ffi.pass_ import AutoDiff, GradientInputSelection
from mnm._lib import tvm
from mnm._lib import relay
from mnm.testing import randn, run_infer_type

def test_conv2d():

    class Model(mnm.Model):
        def build(self):
            self.w, _ = randn((1, 1, 3, 3))

        @mnm.model.trace
        def forward(self, x):
            y = mnm.conv2d(x, self.w)
            z = mnm.relu(y)
            return z

    def expected():
        v_zero = mnm._core.value.IntValue(0)
        v_one = mnm._core.value.IntValue(1)
        v_xd = mnm._core.value.IntValue(224)
        v_wd = mnm._core.value.IntValue(3)
        konst0 = mnm._ffi.ir._make.Constant(
            mnm._core.value.TupleValue([v_zero]))
        int_one = mnm._ffi.ir._make.Constant(v_one)
        konst1 = mnm._ffi.ir._make.Constant(
            mnm._core.value.TupleValue([v_one]))
        constantN = mnm._ffi.ir._make.Constant(None)
        konst_nchw = mnm._ffi.ir._make.Constant(
            mnm._core.value.StringValue("NCHW"))
        konst_oihw = mnm._ffi.ir._make.Constant(
            mnm._core.value.StringValue("OIHW"))

        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        relu = mnm._ffi.op.GetOp("mnm.op.relu")
        relu_dx = mnm._ffi.op.GetOp("mnm.op.relu_dx")
        conv2d_dx = mnm._ffi.op.GetOp("mnm.op.conv2d_dx")
        conv2d_dw = mnm._ffi.op.GetOp("mnm.op.conv2d_dw")

        x = relay.var("x", shape=(1, 1, 224, 224))
        w = relay.var("w", shape=(1, 1, 3, 3))
        dy = relay.var("dy", shape=(1, 1, 222, 222))
        x_shape = mnm._ffi.ir._make.Constant(
            mnm._core.value.TupleValue([v_one, v_one, v_xd, v_xd]))
        w_shape = mnm._ffi.ir._make.Constant(
            mnm._core.value.TupleValue([v_one, v_one, v_wd, v_wd]))

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
        let3 = relay.Let(x3, relay.Call(
            conv2d_dw, [x, constantN, x1, w_shape, konst1, konst0, konst1, int_one]), let4)
        let2 = relay.Let(x2, relay.Call(
            conv2d_dx, [w, constantN, x1, x_shape, konst1, konst0, konst1, int_one]), let3)
        let1 = relay.Let(x1, relay.Call(relu_dx, [constantN, v1, dy]), let2)

        let_ret = relay.Let(ret, relay.Tuple((v1, closure)), ret)
        let_closure = relay.Let(closure, relay.Function([dy], let1), let_ret)

        # forward pass
        letv1 = relay.Let(v1, relay.Call(relu, [v]), let_closure)
        letv = relay.Let(v, relay.Call(
            conv2d_op, [x, w, konst1, konst0, konst1, int_one, konst_nchw, konst_oihw,
                        konst_nchw]), letv1)

        f = relay.Function([x, w], letv)
        return f

    model = Model()
    m_x, _ = randn((1, 1, 224, 224))
    func_before = model._internal(m_x).func
    func_before = run_infer_type(func_before)
    func_before = AutoDiff(func_before)
    func_before = run_infer_type(func_before)
    func_after = GradientInputSelection(func_before)
    func_after = run_infer_type(func_after)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(func_after, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
