# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, invalid-name, attribute-defined-outside-init, no-self-use
# pylint: disable=too-many-locals
import numpy as np
import pytest
import tvm
import raf
from raf._core.ndarray import Symbol
from raf._core.module import IRModule
from raf._core.ir_ext import extended_var
from raf._ffi.pass_ import AutoDiff, ExtractBinding, FromRelay, InferType, LambdaLift
from raf._op import sym as op
from raf.testing import check, randn, run_infer_type
from tvm import relay


def assert_has_type(expr, typ):
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def test_raf_module():
    f1 = relay.GlobalVar("f1")
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


def test_raf_recursive_function():
    f1 = relay.GlobalVar("f1")
    main = relay.GlobalVar("main")

    def get_recursive_mod():
        sb = raf.ir.ScopeBuilder()
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

        n1 = relay.var("n1", ti32)
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


def test_raf_return_function():
    f = relay.GlobalVar("f")
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
    class Model(raf.Model):
        def build(self):
            self.b = raf.array(np.arange(4).reshape([2, 1, 2]), dtype="float32")

        @raf.model.trace
        def forward(self, a):
            c = raf.add(a, self.b)
            x = raf.cos(c)
            y = raf.transpose(x, (0, 2, 1))
            return y

    model = Model()
    m_a, _ = randn((1, 2, 2))
    func = model._internal(m_a).mod["main"]
    func = run_infer_type(func)
    t_1 = relay.TensorType((1, 2, 2))
    t_2 = relay.TensorType((2, 1, 2))
    t_3 = relay.TensorType((2, 2, 2))
    expected_ty = relay.FuncType([t_1, t_2], t_3)
    assert_has_type(func, expected_ty)


def test_any():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, a, b):
            c = raf.add(a, b)
            x = raf.cos(c)
            y = raf.transpose(x, (0, 2, 1))
            return y

    def check_any(shape_a, shape_b, shape_c):
        model = Model()
        a_ty = relay.TensorType(shape_a)
        b_ty = relay.TensorType(shape_b)
        c_ty = relay.TensorType(shape_c)
        a = Symbol.make_var("a", a_ty)
        b = Symbol.make_var("b", b_ty)
        func = model._internal(a, b).mod["main"]
        func = run_infer_type(func)
        expected_ty = relay.FuncType([a_ty, b_ty], c_ty)
        # alpha_equal does not work for Any
        assert str(func.checked_type) == str(expected_ty)

    check_any((relay.Any(), 3, relay.Any()), (2, relay.Any(), 4), (2, 4, 3))
    # broadcast
    check_any((relay.Any(), 3, relay.Any()), (1, relay.Any(), 4), (relay.Any(), 4, 3))


def test_incomplete_call():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, a, b):
            c = raf.add(a, b)
            x = raf.cos(c)
            y = raf.transpose(x, (0, 2, 1))
            return y

    def inc_ty():
        return relay.IncompleteType()

    shape_a = (3, 4, 5)
    a_ty = relay.TensorType(shape_a)
    model = Model()
    a = Symbol.make_var("a", a_ty)
    b = Symbol.make_var("b")
    func = model._internal(a, b).mod["main"]
    func = run_infer_type(func)
    expected_ty = relay.FuncType([a_ty, inc_ty()], inc_ty())
    # alpha_equal does not work for IncompleteType
    assert str(func.checked_type) == str(expected_ty)


def test_gradient_closure():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.transpose(x, (0, 2, 1))
            return y

    def check_backward(shape_x, shape_y):
        model = Model()
        x_ty = relay.TensorType(shape_x)
        y_ty = relay.TensorType(shape_y)
        x = Symbol.make_var("a", x_ty)
        mod = model._internal(x).mod
        mod = InferType()(mod)
        mod = AutoDiff([True])(mod)
        mod = InferType()(mod)
        func = mod["main"]
        bwd_ty = relay.FuncType([y_ty], x_ty)
        expected_ty = relay.FuncType([x_ty], relay.TupleType([y_ty, bwd_ty]))
        assert_has_type(func, expected_ty)

    check_backward((1, 3, 2), (1, 2, 3))


def test_gradient_op():
    x_ty = relay.TensorType((1, 1, 224, 224))
    x = Symbol.make_var("x", x_ty)

    def get_type_func(net):
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
    x_ty = relay.TensorType((1, 1, 224, 224))
    y_ty = relay.TensorType((1, 1, 222, 222))
    w_ty = relay.TensorType((1, 1, 3, 3))

    x = relay.var("x", shape=(1, 1, 224, 224))
    w = relay.var("w", shape=(1, 1, 3, 3))
    dy = relay.var("dy", shape=(1, 1, 222, 222))
    y = relay.var("y", shape=(1, 1, 222, 222))

    def check_conv2d_dx():
        tmp0 = relay.var("tmp0")
        c = raf.ir.op.shape(x)
        let_node = relay.Let(tmp0, raf.ir.op.conv2d_dx(w, y, dy, c, 1, 0, 1, 1), tmp0)
        func = relay.Function([x, w, dy, y], let_node)
        func = run_infer_type(func)
        expected_ty = relay.FuncType([x_ty, w_ty, y_ty, y_ty], x_ty)
        assert_has_type(func, expected_ty)

    check_conv2d_dx()


# pylint: disable=import-outside-toplevel
def test_constant_tensor():
    from raf._ffi.ir.constant import ExtractValue
    from raf._ffi.value import ToTVM

    m_c, n_c = randn((2, 2))
    const_value = raf._core.value.TensorValue.from_numpy(n_c)
    const_value = raf._ffi.ir._make.Constant(const_value)
    func = relay.Function([], const_value)
    func = run_infer_type(func)

    expected_ty = relay.TensorType((2, 2))
    assert str(func.body.checked_type) == str(expected_ty)
    check(m_c, ToTVM(ExtractValue(func.body)).numpy())


def test_constant_tensor_tuple():
    from raf._ffi.ir.constant import ExtractValue
    from raf._ffi.value import ToTVM

    m_c1, n_c1 = randn((2, 2))
    m_c2, n_c2 = randn((3, 3))
    c1_value = raf._core.value.TensorValue.from_numpy(n_c1)
    c2_value = raf._core.value.TensorValue.from_numpy(n_c2)
    const_value = raf._core.value.TupleValue([c1_value, c2_value])
    const_value = raf._ffi.ir._make.Constant(const_value)
    func = relay.Function([], const_value)
    func = run_infer_type(func)

    expected_ty = relay.TupleType([relay.TensorType((2, 2)), relay.TensorType((3, 3))])
    assert str(func.body.checked_type) == str(expected_ty)
    check(m_c1, ToTVM(ExtractValue(func.body)[0]).numpy())
    check(m_c2, ToTVM(ExtractValue(func.body)[1]).numpy())


def test_closure_with_const_args1():
    rand, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = rand

        @raf.model.trace
        def forward(self, x):
            pooled = raf.max_pool2d(x, kernel=(3, 3), stride=1, padding=1)
            return (raf.add(pooled, self.c), x)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = raf._ffi.pass_.InferType()(mod)
    mod = raf._ffi.pass_.FuseTVM()(mod)
    mod = raf._ffi.pass_.InferType()(mod)
    mod = raf._ffi.pass_.ManifestAlloc()(mod)
    # pylint: disable=line-too-long
    #   let %x_2 = fn (%p0: Tensor[(1, 16, 64, 64), float32], %p1: (int32, int32), %p2: (int64,),
    #                  %p3: (int64,), %p4: (int64,), %p5: bool, %p6: bool, %p7: int64,
    #                  %p8: Tensor[(1), float32], Primitive=1) -> Tensor[(1, 16, 64, 64), float32] {
    #     %0 = raf.op.max_pool2d(%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7) /* ty=Tensor[(1, 16, 64, 64), float32] */;
    #     raf.op.add(%0, %p8, nullptr /* ty=() */, nullptr /* ty=() */) /* ty=Tensor[(1, 16, 64, 64), float32] */
    #   };
    #   let %x_3 = (%x, TupleValue([int32(3), int32(3)]) /* ty=(int32, int32) */, TupleValue([int64(1)]) /* ty=(int64,) */, TupleValue([int64(1)]) /* ty=(int64,) */, TupleValue([int64(1)]) /* ty=(int64,) */, bool(0) /* ty=bool */, bool(1) /* ty=bool */, str"NCHW" /* ty=int64 */, %c);
    #   let %x_4 = (%x_1,);
    #   let %x_5 = raf.op.vm.invoke_op(%x_2, %x_3, %x_4);
    # }
    # pylint: enable=line-too-long
    mod = raf._ffi.pass_.InferType()(mod)


def test_closure_with_const_args2():
    # pylint: disable=no-self-use
    # Test ANF before ManifestAlloc.
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.relu(x + x)

    model = Model()
    m_x, _ = randn((3, 4))
    mod = model._internal(m_x).mod
    mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
    mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
    mod = raf._ffi.pass_.FuseTVM()(mod)
    mod = raf._ffi.pass_.ToANormalForm()(mod)
    mod = raf._ffi.pass_.InferType()(mod)


def test_multi_functions():
    # Create a symbolic model and run it
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
    record = model._internal(m_x, m_y)
    mod = record.mod

    # Run AutoDiff to get nested functions
    # The backward function will be lifted
    mod = raf._ffi.pass_.InferType()(mod)
    mod = AutoDiff(record.requires_grads)(mod)

    # Call Lambda lift pass on the RAF module
    lifted_mod = LambdaLift()(mod)
    assert len(lifted_mod.functions) == 2

    # Invoke the backward closure in main
    fwd, bwd, fwd_var, bwd_var = None, None, None, None
    for k, v in lifted_mod.functions.items():
        if k.name_hint == "main":
            fwd_var, fwd = relay.GlobalVar("fwd"), v
        else:
            bwd_var, bwd = k, v
    v, v1, v2, v3 = extended_var("v"), extended_var("v"), extended_var("v"), extended_var("v")
    v4, v5, v6, v7 = extended_var("v"), extended_var("v"), extended_var("v"), extended_var("v")
    x = extended_var("x", shape=(3, 3), dtype="float32")
    y = extended_var("x", shape=(3, 3), dtype="float32")
    dy = extended_var("dy", shape=(3, 3), dtype="float32")
    # pylint: disable=bad-continuation
    body = relay.Let(
        v,
        fwd_var,
        relay.Let(
            v1,
            relay.Call(v, [x, y]),
            relay.Let(
                v2,
                relay.TupleGetItem(v1, 0),
                relay.Let(
                    v3,
                    relay.TupleGetItem(v1, 1),
                    relay.Let(
                        v4,
                        relay.Call(v3, [dy]),
                        relay.Let(
                            v5,
                            relay.TupleGetItem(v4, 0),
                            relay.Let(
                                v6,
                                relay.TupleGetItem(v4, 1),
                                relay.Let(v7, relay.Tuple([v2, v5, v6]), v7),  # forward, dx, dy
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    # pylint: enable=bad-continuation
    func = relay.Function([x, y, dy], body)
    mod = IRModule({fwd_var: fwd, bwd_var: bwd, relay.GlobalVar("main"): func})
    mod = raf._ffi.pass_.InferType()(mod)
    expected_ty = relay.FuncType(
        [relay.TensorType((3, 3)), relay.TensorType((3, 3)), relay.TensorType((3, 3))],
        relay.TupleType(
            [relay.TensorType((3, 3)), relay.TensorType((3, 3)), relay.TensorType((3, 3))]
        ),
    )
    assert mod["main"].checked_type == expected_ty


if __name__ == "__main__":
    pytest.main([__file__])
