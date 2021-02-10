# pylint: disable=protected-access,invalid-name,attribute-defined-outside-init,no-self-use
import pytest
import tvm
from tvm import relay
import mnm
from mnm.testing import randn


def test_basic():
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    def expected(shape):
        # pylint: disable=too-many-locals,
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        sum_op = mnm._ffi.op.GetOp("mnm.op.sum")
        get_reduce_axis_op = mnm._ffi.op.GetOp("mnm.op.get_reduce_axis")
        get_kept_dims_op = mnm._ffi.op.GetOp("mnm.op.get_kept_dims")
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        dy = relay.var("dy")
        a1 = relay.var("a1")
        x1 = relay.var("x1")
        x2 = relay.var("x2")
        x3 = relay.var("x3")
        x4 = relay.var("x4")
        x5 = relay.var("x5")
        x6 = relay.var("x6")
        gradient = relay.var("gradient")
        ret = relay.var("ret")

        let9 = relay.Let(ret, relay.Tuple([a1, gradient]), ret)
        let8 = relay.Let(gradient, relay.Tuple([x3, x6]), let9)
        let7 = relay.Let(x6, relay.Call(sum_op, [dy, x4, x5]), let8)
        let6 = relay.Let(x5, relay.Call(get_kept_dims_op, [dy, y]), let7)
        let5 = relay.Let(x4, relay.Call(get_reduce_axis_op, [dy, y]), let6)
        let4 = relay.Let(x3, relay.Call(sum_op, [dy, x1, x2]), let5)
        let3 = relay.Let(x2, relay.Call(get_kept_dims_op, [dy, x]), let4)
        let2 = relay.Let(x1, relay.Call(get_reduce_axis_op, [dy, x]), let3)
        let1 = relay.Let(a1, relay.Call(add_op, [x, y, default, default]), let2)
        return relay.Function([x, y, dy], let1)

    shape = (4, 5)
    model = Add()
    model.train_mode()
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)
    m_x.requires_grad = True
    m_y.requires_grad = True
    record = model._internal(m_x, m_y)
    func = record.func
    func = mnm._ffi.pass_.AutoDiff(func, record.requires_grads)
    inlined_func = mnm._ffi.pass_.InlineBackward(func)
    assert tvm.ir.structural_equal(inlined_func, expected(shape))


def test_no_backward():
    class Model1(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    # model that returns a tuple
    class Model2(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.split(mnm.add(x, y), 2)

    # Get a Relay func
    shape = (4, 5)
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)

    model1 = Model1()
    func = model1._internal(m_x, m_y).func
    inlined_func = mnm._ffi.pass_.InlineBackward(func)
    assert tvm.ir.structural_equal(inlined_func, func)

    model2 = Model2()
    func = model2._internal(m_x, m_y).func
    inlined_func = mnm._ffi.pass_.InlineBackward(func)
    assert tvm.ir.structural_equal(inlined_func, func)


if __name__ == "__main__":
    pytest.main([__file__])
