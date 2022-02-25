# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name,protected-access,too-many-locals,attribute-defined-outside-init
import numpy as np
import pytest
import raf
import tvm
from tvm import relay
from raf._ffi.pass_ import InferType, AutoDiff, FromRelay, LiftBranchBody
from raf._ffi.pass_ import LambdaLift, FlattenClosure, InlineBackward
from raf.ir import RAFSequential, ScopeBuilder
from raf.model import BatchNorm
from raf.model.trace import _get_func_inputs
from raf.testing import get_testable_devices, randn, check, utils, one_hot_torch


def ad_passes(mod, requires_grads=None):
    if requires_grads is None:
        requires_grads = []
    seq = RAFSequential(
        [
            InferType(),
            LambdaLift(),
            InferType(),
            FlattenClosure(),
            InferType(),
            LiftBranchBody(),
            InferType(),
            AutoDiff(requires_grads),
            InferType(),
        ]
    )
    return seq(mod)


def vm_passes(mod, requires_grads=None):
    if requires_grads is None:
        requires_grads = []
    mod = ad_passes(mod, requires_grads)
    mod = InlineBackward()(mod)
    return mod


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("shape", [[3, 3], [4, 4]])
def test_add_to(shape, device):
    class Add(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return raf.add(x, x)

    model = Add()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, device=device)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 2 * n_dy
    check(m_dx, n_dx)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("planes", [16])
def test_batch_norm(device, planes):
    class BatchNormLoss(raf.Model):
        def build(self, planes):
            self.bn = BatchNorm(planes)

        @raf.model.trace
        def forward(self, x, y_true):
            out = self.bn(x)
            y_pred = raf.batch_flatten(out)
            loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
            return loss

    model = BatchNormLoss(planes)
    m_x, _ = randn([5, planes, 32, 32], device=device)
    m_x.requires_grad = True
    m_ytrue, _ = one_hot_torch(5, num_classes=16384, device="cpu")
    m_ytrue.requires_grad = False

    record = model._internal(m_x, m_ytrue)
    mod = record.mod
    ad_mod = ad_passes(mod, record.requires_grads)

    assert len(ad_mod.get_global_vars()) == 1
    fwd_type = relay.TupleType(
        [
            relay.TensorType((1,)),
            relay.TensorType((planes,)),
            relay.TensorType((planes,)),
        ]
    )
    bwd_type = relay.FuncType(
        [fwd_type],
        relay.TupleType(
            [
                relay.TensorType((5, planes, 32, 32), "float32"),
                relay.TensorType((), "int64"),
                relay.TensorType((planes,), "float32"),
                relay.TensorType((), "int64"),
                relay.TensorType((), "int64"),
                relay.TensorType((planes,), "float32"),
            ]
        ),
    )
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_dy_0 = raf.array(np.array([1.0], dtype="float32"), device=device)
    m_dy_1, _ = randn((planes,), device=device)
    m_dy_2, _ = randn((planes,), device=device)
    inputs = _get_func_inputs(record, [], {"x": m_x, "y_true": m_ytrue}, get_handle=False)
    inputs += [(m_dy_0, m_dy_1, m_dy_2)]
    vm_executor = utils.get_vm_executor(mod, device, pass_seq=vm_passes)
    vm_executor(*inputs)


@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize(
    "shape",
    [
        [
            3,
        ],
        [
            4,
        ],
    ],
)
def test_no_grad1(shape, device):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y, z):  # pylint: disable=no-self-use
            indices = raf.add(y, z)
            indices = raf.subtract(indices, z)
            indices = raf.add(indices, z)
            return raf.take(x, indices, axis=0)

    model = Model()
    m_x, n_x = randn(shape, device=device, requires_grad=True)
    m_y = raf.array(
        [
            1,
        ],
        dtype="int64",
        device=device,
    )
    m_z = raf.array(
        [
            1,
        ],
        dtype="int64",
        device=device,
    )
    m_out = model(m_x, m_y, m_z)  # m_out = m_x[2]
    m_dout, n_dout = randn(
        [
            1,
        ],
        device=device,
    )
    m_out.backward(m_dout)
    m_dx = m_x.grad
    n_dx = np.zeros_like(n_x)
    n_dx[2] = n_dout[0]
    check(m_dx, n_dx)


@pytest.mark.parametrize("device", get_testable_devices())
def test_no_grad2(device):
    shape = [3, 2]
    dtype = "float32"

    def expected():
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        a1 = relay.var("a1")
        closure = relay.var("adjoint_closure")
        dy = relay.var("dy", shape=(3, 3))
        x1 = relay.var("x")
        x2 = relay.var("x")
        ret = relay.var("ret")
        inner_let2 = relay.Let(
            x2,
            relay.Tuple([x1, raf._ffi.ir._make.Constant(raf._core.value.NoGradValue())]),
            x2,
        )
        inner_let1 = relay.Let(x1, raf.ir.op.matmul(dy, y), inner_let2)
        let3 = relay.Let(ret, relay.Tuple([a1, closure]), ret)
        let2 = relay.Let(closure, relay.Function([dy], body=inner_let1), let3)
        let1 = relay.Let(a1, raf.ir.op.matmul_nt(x, y), let2)
        func = relay.Function([x, y], let1)
        mod = tvm.IRModule()
        mod["main"] = func
        mod = InferType()(mod)
        return mod["main"]

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):  # pylint: disable=no-self-use
            return raf.matmul_nt(x, y)

    model = Model()
    # forward
    m_x, _ = randn(shape, dtype=dtype, device=device)
    m_y, _ = randn(shape, dtype=dtype, device=device)
    m_x.requires_grad = True
    m_y.requires_grad = False

    m_record = model._internal(m_x, m_y)
    # backward
    m_mod = InferType()(m_record.mod)
    m_mod = AutoDiff(m_record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    assert tvm.ir.structural_equal(m_mod["main"], expected())


@pytest.mark.parametrize("device", get_testable_devices())
def test_basic(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.erf(a)
        c = relay.nn.relu(b)
        out = c

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    fwd_type = relay.TensorType((1, 100))
    bwd_type = relay.FuncType([fwd_type], fwd_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_concatenate(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.erf(a)
        c = relay.concatenate([a, b], axis=0)
        d = relay.nn.relu(c)
        out = d

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TensorType((2, 100))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((2, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_split(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 102), dtype="float32")

        a = relay.tanh(x)
        b = relay.split(a, indices_or_sections=3, axis=1)
        c = relay.nn.relu(b[0])
        d = relay.erf(b[1])
        e = relay.erf(b[2])
        f = relay.add(c, d)
        g = relay.add(e, f)
        out = g

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 102))
    fwd_type = relay.TensorType((1, 34))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 102), device=device)
    m_dy, _ = randn((1, 34), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_split_unused_output(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.split(a, indices_or_sections=2, axis=1)
        c = relay.nn.relu(b[0])
        out = c

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)

    ad_mod = ad_passes(mod)
    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TensorType((1, 50))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 50), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_fanout(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.nn.relu(a)
        c = relay.erf(a)
        d = relay.add(b, c)
        out = d

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TensorType((1, 100))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_split_concat(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.split(a, indices_or_sections=2, axis=1)
        c = relay.concatenate(b, axis=1)
        d = relay.erf(c)
        out = d

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TensorType((1, 100))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_split_with_fanout(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.split(a, indices_or_sections=2, axis=1)
        c = relay.nn.relu(b[0])
        d = relay.erf(b[0])
        e = relay.add(c, b[1])
        f = relay.add(d, b[1])
        g = relay.concatenate([e, f], axis=1)
        out = g

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TensorType((1, 100))
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_dy)


@pytest.mark.parametrize("device", get_testable_devices())
def test_concatenate_fanout(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")

        a = relay.Tuple([x, y])
        b = relay.concatenate(a, axis=1)
        c = relay.tanh(b)
        d = relay.concatenate(a, axis=1)
        e = relay.nn.relu(d)
        f = relay.Tuple([c, e])
        out = f

        func = relay.Function([x, y], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_field_type = relay.TensorType((1, 100))
    fwd_field_type = relay.TensorType((1, 200))
    var_type = relay.TupleType([var_field_type, var_field_type])
    fwd_type = relay.TupleType([fwd_field_type, fwd_field_type])
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_y, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 200), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, m_y, [m_dy, m_dy])


@pytest.mark.parametrize("device", get_testable_devices())
def test_tuple_outputs(device):
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.nn.relu(x)
        c = relay.Tuple([a, b])
        out = c

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TupleType([var_type, var_type])
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    m_x, _ = randn((1, 100), device=device)
    m_dy, _ = randn((1, 100), device=device)
    vm_executor = utils.get_vm_executor(mod, "cpu", pass_seq=vm_passes)
    vm_executor(m_x, [m_dy, m_dy])


def test_nested_tuple_outputs():
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")

        a = relay.tanh(x)
        b = relay.nn.relu(x)
        c = relay.Tuple([a, b])
        d = relay.erf(x)
        e = relay.Tuple([c, d])
        out = e

        func = relay.Function([x], out)
        mod["main"] = func
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 1
    var_type = relay.TensorType((1, 100))
    fwd_type = relay.TupleType([relay.TupleType([var_type, var_type]), var_type])
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # TODO (@janimesh) - List of lists is not supported in vm_executor
    # Run via VM to ensure that the generated mod can be executed
    # m_x, _ = randn((1, 100), device=device)
    # m_dy, _ = randn((1, 100), device=device)
    # vm_executor = \
    #   utils.get_vm_executor(model, 'cpu', pass_seq=vm_passes)
    # m_out = vm_executor(m_x, [[m_dy, m_dy], m_dy])


def test_basic_function():
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(1, 100), dtype="float32")
        a = relay.tanh(x)
        f1_out = a
        f1 = relay.Function([x], f1_out)
        f1_gvar = relay.GlobalVar("f1")
        mod[f1_gvar] = f1
        mod = relay.transform.InferType()(mod)

        y = relay.var("y", shape=(1, 100), dtype="float32")
        a = relay.erf(y)
        b = f1_gvar(a)
        c = relay.nn.relu(b)
        out = c

        func = relay.Function([y], out)
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 2
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # Run via VM to ensure that the generated mod can be executed
    # TODO (@janimesh) - Fails because of InlineBackward
    # m_x, _ = randn((1, 100), device=device)
    # m_dy, _ = randn((1, 100), device=device)
    # vm_executor = utils.get_vm_executor(model, 'cpu', pass_seq=vm_passes)
    # m_out = vm_executor(m_x, m_dy)


def test_basic_function_with_multiple_vars():
    def get_mod():
        mod = tvm.IRModule()

        x = relay.var("x", shape=(1, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")
        a = relay.tanh(x)
        b = relay.nn.relu(y)
        c = relay.add(a, b)
        f1_out = c
        f1 = relay.Function([x, y], f1_out)
        f1_gvar = relay.GlobalVar("f1")
        mod[f1_gvar] = f1
        mod = relay.transform.InferType()(mod)

        p = relay.var("p", shape=(1, 100), dtype="float32")
        q = relay.var("q", shape=(1, 100), dtype="float32")
        a = relay.nn.relu(p)
        b = f1_gvar(a, q)
        out = b

        func = relay.Function([p, q], out)
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 2
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], relay.TupleType([var_type, var_type]))
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # TODO (@janimesh) - List of lists is not supported in vm_executor
    # Run via VM to ensure that the generated mod can be executed


def test_function_fanout():
    def get_mod():
        mod = tvm.IRModule()

        x = relay.var("x", shape=(1, 100), dtype="float32")
        a = relay.tanh(x)
        f1_out = a
        f1 = relay.Function([x], f1_out)
        f1_gvar = relay.GlobalVar("f1")
        mod[f1_gvar] = f1
        mod = relay.transform.InferType()(mod)

        p = relay.var("p", shape=(1, 100), dtype="float32")
        a = relay.nn.relu(p)
        b = f1_gvar(a)
        c = relay.tanh(b)
        d = relay.erf(b)
        e = relay.add(c, d)
        out = e

        func = relay.Function([p], out)
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 2
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type


def test_basic_if():
    main = relay.GlobalVar("main")

    def get_mod():
        sb = ScopeBuilder()
        mod = tvm.IRModule()

        # Recursive function f
        ti32 = relay.scalar_type("int32")
        n = relay.var("n", ti32)
        x = relay.var("x", shape=(1, 100), dtype="float32")
        with sb.if_scope(relay.equal(n, relay.const(0, ti32))):
            sb.ret(relay.tanh(x))
        with sb.else_scope():
            sb.ret(relay.erf(x))
        mod[main] = relay.Function([n, x], sb.get())
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 3
    ti32 = relay.scalar_type("int32")
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], relay.TupleType([ti32, var_type]))
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # TODO (@janimesh) - List of lists is not supported in vm_executor
    # Run via VM to ensure that the generated mod can be executed


def test_if_with_multiple_vars():
    main = relay.GlobalVar("main")

    def get_mod():
        sb = ScopeBuilder()
        mod = tvm.IRModule()

        # Recursive function f
        ti32 = relay.scalar_type("int32")
        n = relay.var("n", ti32)
        x = relay.var("x", shape=(1, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")
        with sb.if_scope(relay.equal(n, relay.const(0, ti32))):
            sb.ret(relay.add(relay.tanh(x), relay.tanh(y)))
        with sb.else_scope():
            sb.ret(relay.add(relay.erf(x), relay.erf(y)))
        mod[main] = relay.Function([n, x, y], sb.get())
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 3
    ti32 = relay.scalar_type("int32")
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], relay.TupleType([ti32, var_type, var_type]))
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # TODO (@janimesh) - List of lists is not supported in vm_executor
    # Run via VM to ensure that the generated mod can be executed


def test_recursive_with_if():
    f1 = relay.GlobalVar("f1")  # pylint: disable=invalid-name
    main = relay.GlobalVar("main")

    def get_mod():
        sb = ScopeBuilder()
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

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 4
    ti32 = relay.scalar_type("int32")
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], relay.TupleType([ti32, var_type]))
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type

    # TODO (@janimesh) - List of lists is not supported in vm_executor
    # Run via VM to ensure that the generated mod can be executed


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

    def get_mod():
        sb = ScopeBuilder()
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

        loop_call = relay.Call(let, [relay.const(0, ti32), y])
        body = relay.TupleGetItem(loop_call, 1)
        mod["main"] = relay.Function([y], body)
        mod = relay.transform.InferType()(mod)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    ad_mod = ad_passes(mod)

    assert len(ad_mod.get_global_vars()) == 4
    var_type = relay.TensorType((1, 100))
    fwd_type = var_type
    bwd_type = relay.FuncType([fwd_type], var_type)
    ret_type = relay.TupleType([fwd_type, bwd_type])
    assert ad_mod["main"].ret_type == ret_type


def test_simplify_sum():
    # Get a Relay func
    def get_mod():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(10, 100), dtype="float32")
        y = relay.var("y", shape=(1, 100), dtype="float32")
        out = relay.add(x, y)
        mod["main"] = relay.Function([x, y], out)
        return mod

    tvm_mod = get_mod()
    mod = FromRelay()(tvm_mod)
    seq = RAFSequential([InferType(), AutoDiff([])])
    mod = seq(mod)

    # Ensure that there is only one sum operator
    sum_ops = list()
    find_sum = lambda x: sum_ops.append(isinstance(x, tvm.relay.Call) and x.op.name == "raf.op.sum")
    tvm.relay.analysis.post_order_visit(mod["main"], find_sum)
    assert len(list(filter(lambda x: x, sum_ops))) == 1


if __name__ == "__main__":
    pytest.main([__file__])
