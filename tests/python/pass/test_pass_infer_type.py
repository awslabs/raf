# pylint: disable=protected-access
import numpy as np
import pytest
import mnm
from mnm._core.ndarray import Symbol
from mnm._ffi.pass_ import InferType, AutoDiff, ExtractBinding
from mnm._op import sym as op
from tvm import relay


def assert_has_type(expr, typ):
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def run_infer_type(func):
    # pylint: disable=protected-access
    mod = mnm._ffi.ir._make.Module({relay.GlobalVar("main"): func})
    mod = InferType(mod)
    return mod['main']


def test_model_params():
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            self.b = mnm.array(np.arange(4).reshape(
                [2, 1, 2]), dtype='float32')

        @mnm.model.trace
        def forward(self, a):
            c = mnm.add(a, self.b)
            x = mnm.cos(c)
            y = mnm.transpose(x, (0, 2, 1))
            return y

    model = Model()
    m_a, _ = randn((1, 2, 2))
    func = model._internal(m_a).func
    func = run_infer_type(func)
    t_1 = relay.TensorType((1, 2, 2))
    t_2 = relay.TensorType((2, 1, 2))
    t_3 = relay.TensorType((2, 2, 2))
    expected_ty = relay.FuncType([t_1, t_2], t_3)
    assert_has_type(func, expected_ty)


def test_any():
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, a, b):
            c = mnm.add(a, b)
            x = mnm.cos(c)
            y = mnm.transpose(x, (0, 2, 1))
            return y

    def check_any(shape_a, shape_b, shape_c):
        model = Model()
        a_ty = relay.TensorType(shape_a)
        b_ty = relay.TensorType(shape_b)
        c_ty = relay.TensorType(shape_c)
        a = Symbol.make_var('a', a_ty)
        b = Symbol.make_var('b', b_ty)
        func = model._internal(a, b).func
        func = run_infer_type(func)
        expected_ty = relay.FuncType([a_ty, b_ty], c_ty)
        # alpha_equal does not work for Any
        assert str(func.checked_type) == str(expected_ty)

    check_any((relay.Any(), 3, relay.Any()), (2, relay.Any(), 4), (2, 4, 3))
    # broadcast
    check_any((relay.Any(), 3, relay.Any()),
              (1, relay.Any(), 4), (relay.Any(), 4, 3))


def test_incomplete_call():
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, a, b):
            c = mnm.add(a, b)
            x = mnm.cos(c)
            y = mnm.transpose(x, (0, 2, 1))
            return y

    def inc_ty():
        return relay.IncompleteType()

    shape_a = (3, 4, 5)
    a_ty = relay.TensorType(shape_a)
    model = Model()
    a = Symbol.make_var('a', a_ty)
    b = Symbol.make_var('b')
    func = model._internal(a, b).func
    func = run_infer_type(func)
    expected_ty = relay.FuncType([a_ty, inc_ty()], inc_ty())
    # alpha_equal does not work for IncompleteType
    assert str(func.checked_type) == str(expected_ty)


def test_gradient_closure():
    class Model(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self):
            pass

        # pylint: disable=no-self-use
        @mnm.model.trace
        def forward(self, x):
            y = mnm.transpose(x, (0, 2, 1))
            return y

    def check_backward(shape_x, shape_y):
        model = Model()
        x_ty = relay.TensorType(shape_x)
        y_ty = relay.TensorType(shape_y)
        x = Symbol.make_var('a', x_ty)
        func = model._internal(x).func
        func = run_infer_type(func)
        func = AutoDiff(func)
        func = run_infer_type(func)
        bwd_ty = relay.FuncType([y_ty], x_ty)
        expected_ty = relay.FuncType([x_ty], relay.TupleType([y_ty, bwd_ty]))
        assert_has_type(func, expected_ty)

    check_backward((1, 3, 2), (1, 2, 3))


def test_gradient_op():
    x_ty = relay.TensorType((1, 1, 224, 224))
    x = Symbol.make_var("x", x_ty)

    def get_type_func(net):
        # pylint: disable=protected-access
        body = net._Symbol__handle
        body = ExtractBinding(body)
        func = relay.Function(relay.analysis.free_vars(body), body)
        func = run_infer_type(func)
        return func

    def check_relu_dx():
        dx = op.relu_dx(x, x, x)
        expected_ty = relay.FuncType([x_ty], x_ty)
        func = get_type_func(dx)
        assert_has_type(func, expected_ty)

    check_relu_dx()


def test_shape_op():
    # pylint: disable=protected-access
    shape = mnm._ffi.op.GetOp("mnm.op.shape")
    conv2d_dx = mnm._ffi.op.GetOp("mnm.op.conv2d_dx")

    konst0 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(0))
    konst1 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(1))

    x_ty = relay.TensorType((1, 1, 224, 224))
    y_ty = relay.TensorType((1, 1, 222, 222))
    w_ty = relay.TensorType((1, 1, 3, 3))

    x = relay.var("x", shape=(1, 1, 224, 224))
    w = relay.var("w", shape=(1, 1, 3, 3))
    dy = relay.var("dy", shape=(1, 1, 222, 222))
    y = relay.var("y", shape=(1, 1, 222, 222))

    def check_conv2d_dx():
        tmp0 = relay.var("tmp0")
        c = relay.Call(shape, [x])
        let_node = relay.Let(tmp0, relay.Call(
            conv2d_dx, [w, y, dy, c, konst1, konst0, konst1, konst1]), tmp0)
        func = relay.Function([x, w, dy, y], let_node)
        func = run_infer_type(func)
        expected_ty = relay.FuncType([x_ty, w_ty, y_ty, y_ty], x_ty)
        assert_has_type(func, expected_ty)

    check_conv2d_dx()


# pylint: disable=import-outside-toplevel, protected-access
def test_constant_tensor():
    from mnm._ffi.ir.constant import ExtractValue
    from mnm._ffi.value import ToTVM
    m_c, n_c = randn((2, 2))
    const_value = mnm._core.value.TensorValue.from_numpy(n_c)
    const_value = mnm._ffi.ir._make.Constant(const_value)
    func = relay.Function([], const_value)
    func = run_infer_type(func)

    expected_ty = relay.TensorType((2, 2))
    assert str(func.body.checked_type) == str(expected_ty)
    check(m_c, ToTVM(ExtractValue(func.body)).asnumpy())


def test_constant_tensor_tuple():
    from mnm._ffi.ir.constant import ExtractValue
    from mnm._ffi.value import ToTVM
    m_c1, n_c1 = randn((2, 2))
    m_c2, n_c2 = randn((3, 3))
    c1_value = mnm._core.value.TensorValue.from_numpy(n_c1)
    c2_value = mnm._core.value.TensorValue.from_numpy(n_c2)
    const_value = mnm._core.value.TupleValue([c1_value, c2_value])
    const_value = mnm._ffi.ir._make.Constant(const_value)
    func = relay.Function([], const_value)
    func = run_infer_type(func)

    expected_ty = relay.TupleType([relay.TensorType((2, 2)), relay.TensorType((3, 3))])
    assert str(func.body.checked_type) == str(expected_ty)
    check(m_c1, ToTVM(ExtractValue(func.body)[0]).asnumpy())
    check(m_c2, ToTVM(ExtractValue(func.body)[1]).asnumpy())


if __name__ == "__main__":
    pytest.main([__file__])
