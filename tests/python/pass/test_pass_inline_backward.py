# pylint: disable=protected-access,invalid-name,attribute-defined-outside-init,no-self-use
import pytest
import tvm
from tvm import relay
import mnm
from mnm.ir import MNMSequential
from mnm.testing import randn


def test_basic():
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return mnm.add(x, y)

    def expected(shape):
        # pylint: disable=too-many-locals
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        dy = relay.var("dy", shape=shape)
        a1 = relay.var("a1")
        gradient = relay.var("gradient")
        ret = relay.var("ret")

        let3 = relay.Let(ret, relay.Tuple([a1, gradient]), ret)
        let2 = relay.Let(gradient, relay.Tuple([dy, dy]), let3)
        let1 = relay.Let(a1, mnm.ir.op.add(x, y), let2)
        func = relay.Function([x, y, dy], let1)
        mod = tvm.IRModule.from_expr(func)
        mod = mnm._ffi.pass_.InferType()(mod)
        return mod["main"]

    shape = (4, 5)
    model = Add()
    model.train_mode()
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)
    m_x.requires_grad = True
    m_y.requires_grad = True
    record = model._internal(m_x, m_y)
    mod = record.mod
    seq = MNMSequential(
        [
            mnm._ffi.pass_.InferType(),
            mnm._ffi.pass_.AutoDiff(record.requires_grads),
            mnm._ffi.pass_.InlineBackward(),
            mnm._ffi.pass_.InferType(),
        ]
    )
    mod = seq(mod)
    inlined_func = mod["main"]
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
    mod = model1._internal(m_x, m_y).mod
    func = mod["main"]
    inlined_func = mnm._ffi.pass_.InlineBackward()(mod)["main"]
    assert tvm.ir.structural_equal(inlined_func, func)

    model2 = Model2()
    mod = model2._internal(m_x, m_y).mod
    func = mod["main"]
    inlined_func = mnm._ffi.pass_.InlineBackward()(mod)["main"]
    assert tvm.ir.structural_equal(inlined_func, func)


if __name__ == "__main__":
    pytest.main([__file__])
