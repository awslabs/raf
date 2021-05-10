# pylint: disable=protected-access
import numpy as np
import pytest
import tvm
import mnm
from mnm._core.ndarray import Symbol
from mnm._ffi.pass_ import AutoDiff, ExtractBinding, FromRelay, InferType
from mnm._op import sym as op
from mnm.testing import check, randn, run_infer_type
from tvm import relay


def assert_has_type(expr, typ):
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def test_mnm_module():
    f1 = relay.GlobalVar("f1")  # pylint: disable=invalid-name
    main = relay.GlobalVar("main")
    def get_tvm_mod():
        x = relay.var("x", shape=(1, 100))
        tanh = relay.tanh(x)
        out = tanh
        tvm_mod = tvm.IRModule()
        tvm_mod[f1] = relay.Function([x], out)
        tvm_mod = relay.transform.InferType()(tvm_mod)

        y = relay.var("y", shape=(1, 100))
        out = f1(y)
        tvm_mod[main] = relay.Function([y], out)
        tvm_mod = relay.transform.InferType()(tvm_mod)
        return tvm_mod

    tvm_mod = get_tvm_mod()
    mod = FromRelay()(tvm_mod)
    mod = InferType()(mod)

    t_1 = relay.TensorType((1, 100))
    t_2 = relay.TensorType((1, 100))
    expected_ty = relay.FuncType([t_1], t_2)
    assert mod[f1].checked_type == expected_ty
    assert mod[main].checked_type == expected_ty


def test_mnm_recursive_function():
    f1 = relay.GlobalVar("f1")  # pylint: disable=invalid-name
    main = relay.GlobalVar("main")
    def get_recursive_mod():
        sb = relay.ScopeBuilder()  # pylint: disable=invalid-name
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

    t_0 = relay.scalar_type(dtype="int32")
    t_1 = relay.TensorType((1, 100))
    t_2 = relay.TensorType((1, 100))
    expected_ty = relay.FuncType([t_0, t_1], t_2)
    assert mod[f1].checked_type == expected_ty
    assert mod[main].checked_type == expected_ty


def test_mnm_return_function():
    f = relay.GlobalVar("f")  # pylint: disable=invalid-name
    main = relay.GlobalVar("main")
    def get_tvm_mod():
        tvm_mod = tvm.IRModule()

        y = relay.var("y", shape=(1, 100))
        z = relay.var("z", shape=(1, 100))
        tanh = relay.add(y, z)
        closure = relay.Function([y], tanh)
        tvm_mod[f] = relay.Function([z], closure)
        tvm_mod = relay.transform.InferType()(tvm_mod)

        x = relay.var("x", shape=(1, 100))
        a = relay.var("a", shape=(1, 100))
        closure = f(x)
        closure_call = relay.Call(closure, [a])
        tvm_mod[main] = relay.Function([x, a], closure_call)
        tvm_mod = relay.transform.InferType()(tvm_mod)
        return tvm_mod

    tvm_mod = get_tvm_mod()
    mod = FromRelay()(tvm_mod)
    mod = InferType()(mod)

    t_1 = relay.TensorType((1, 100))
    assert mod[main].ret_type == t_1


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
    func = model._internal(m_a).mod['main']
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
        func = model._internal(a, b).mod['main']
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
    func = model._internal(a, b).mod['main']
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
        mod = model._internal(x).mod
        mod = InferType()(mod)
        mod = AutoDiff([True])(mod)
        mod = InferType()(mod)
        func = mod['main']
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
        body = ExtractBinding(body, [])
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

def test_closure_with_const_args1():
    # pylint: disable=attribute-defined-outside-init
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand

        @mnm.model.trace
        def forward(self, x):
            pooled = mnm.max_pool2d(x, kernel=(3, 3), stride=1, padding=1)
            return (mnm.add(pooled, self.c), x)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = mnm._ffi.pass_.FuseOps(3)(mod)
    mod = mnm._ffi.pass_.InferType()(mod)
    mod = mnm._ffi.pass_.ManifestAlloc()(mod)
    # pylint: disable=line-too-long
    #   let %x_2 = fn (%p0: Tensor[(1, 16, 64, 64), float32], %p1: (int32, int32), %p2: (int64,),
    #                  %p3: (int64,), %p4: (int64,), %p5: bool, %p6: bool, %p7: int64,
    #                  %p8: Tensor[(1), float32], Primitive=1) -> Tensor[(1, 16, 64, 64), float32] {
    #     %0 = mnm.op.max_pool2d(%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7) /* ty=Tensor[(1, 16, 64, 64), float32] */;
    #     mnm.op.add(%0, %p8, nullptr /* ty=() */, nullptr /* ty=() */) /* ty=Tensor[(1, 16, 64, 64), float32] */
    #   };
    #   let %x_3 = (%x, TupleValue([int32(3), int32(3)]) /* ty=(int32, int32) */, TupleValue([int64(1)]) /* ty=(int64,) */, TupleValue([int64(1)]) /* ty=(int64,) */, TupleValue([int64(1)]) /* ty=(int64,) */, bool(0) /* ty=bool */, bool(1) /* ty=bool */, str"NCHW" /* ty=int64 */, %c);
    #   let %x_4 = (%x_1,);
    #   let %x_5 = mnm.op.vm.invoke_op(%x_2, %x_3, %x_4);
    # }
    # pylint: enable=line-too-long
    mod = mnm._ffi.pass_.InferType()(mod)

def test_closure_with_const_args2():
    # pylint: disable=no-self-use
    # Test ANF before ManifestAlloc.
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.relu(x + x)

    model = Model()
    m_x, _ = randn((3, 4))
    mod = model._internal(m_x).mod
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    mod = mnm._ffi.pass_.FuseOps(3)(mod)
    mod = mnm._ffi.pass_.ToANormalForm()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)


if __name__ == "__main__":
    pytest.main([__file__])
