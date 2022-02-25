# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, no-self-use, attribute-defined-outside-init
# pylint: disable=too-many-locals, too-many-arguments
import pytest
import numpy as np
import raf
import tvm
from raf._ffi import pass_
from raf._core.ir_ext import ExtendedVar
from raf.ir import ScopeBuilder
from raf.model.nn import BatchNorm
from raf.model.trace import trace_mutate_attr
from raf.testing import get_testable_devices, compile_vm_model, run_vm_model, check, randn
from tvm import relay


def extract_vars(body):
    if isinstance(body, relay.Var):
        return []
    assert isinstance(body, relay.Let)
    variables = extract_vars(body.body)
    if isinstance(body.value, relay.If):
        tvs = extract_vars(body.value.true_branch)
        fvs = extract_vars(body.value.false_branch)
        variables = tvs + fvs + variables
    variables = [body.var] + variables
    return variables


def optimize(mod):
    mod = pass_.ToGraphNormalForm()(mod)
    mod = pass_.ToBasicBlockNormalForm()(mod)
    mod = pass_.ToANormalForm()(mod)
    mod = pass_.InferType()(mod)
    mod = pass_.InplaceUpdate()(mod)
    return mod


def lower(model, args):
    mod = model._internal(*args).mod
    return optimize(mod)


def checkir(variables, alias):
    for var in alias:
        assert var in variables
    for var in variables:
        assert isinstance(var, relay.Var)
        var = ExtendedVar(var)
        may_share = var.may_share
        if var not in alias:
            assert may_share is None
        else:
            assert may_share == alias[var]


def test_bn():
    shape = (2, 3, 4, 5)
    dtype = "float32"
    data = raf.array(np.ones(shape), dtype=dtype)

    class Test1(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @raf.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = raf.relu(x)
            return x

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) -> (Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) {
    #   let %a1 = raf.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a2 = %a1.0;
    #   let %a3 = raf.op.relu(%a2) /* ty=Tensor[(2, 3, 4, 5), float32] */;
    #   let %a4 = %a1.1;
    #   let %a5 = %a1.2;
    #   let %a6 = (%a3, %a4, %a5);
    #   %a6
    # }
    # pylint: enable=line-too-long
    model = Test1(num_features=3)
    func = lower(model, [data])["main"]
    variables = extract_vars(func.body)
    running_mean = func.params[2]
    running_var = func.params[3]
    alias = {
        variables[3]: running_mean,
        variables[4]: running_var,
    }
    checkir(variables, alias)

    class Test2(raf.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @raf.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = raf.relu(x)
            # typically batch_norm is not used in this way:
            # two batch_norm generally correspond to two BatchNorm models.
            # for test only here.
            x = self.batch_norm(x)
            return x

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) -> (Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) {
    #   let %a1 = raf.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a2 = %a1.0;
    #   let %a3 = raf.op.relu(%a2) /* ty=Tensor[(2, 3, 4, 5), float32] */;
    #   let %a4 = %a1.1;
    #   let %a5 = %a1.2;
    #   let %a6 = raf.op.batch_norm_train(%a3, %a4, %a5, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a7 = %a6.0;
    #   let %a8 = %a6.1;
    #   let %a9 = %a6.2;
    #   let %a10 = (%a7, %a8, %a9);
    #   %a10
    # }
    # pylint: enable=line-too-long
    model = Test2(num_features=3)
    func = lower(model, [data])["main"]
    variables = extract_vars(func.body)
    running_mean = func.params[2]
    running_var = func.params[3]
    alias = {
        variables[3]: running_mean,
        variables[4]: running_var,
        variables[7]: running_mean,
        variables[8]: running_var,
    }
    checkir(variables, alias)


@pytest.mark.parametrize("device", get_testable_devices())
def test_grad(device):
    class Model(raf.Model):
        def build(self, shape):
            self.shape = shape
            self.reset()

        def reset(self):
            self.x = raf.array(np.random.randn(*self.shape), device=device)

        @raf.model.trace
        def forward(self):
            return raf.relu(self.x)

    # fn (%dy: Tensor[(2, 3, 4), float64], %model.x: Tensor[(2, 3, 4), float64]) {
    #     let %a1 = raf.op.relu(%model.x);
    #     let %a2 = raf.op.relu_dx(%model.x, %a1, %dy);
    #     let %a3 = raf.op.subtract(%model.x, %a2, -114514, -114514);
    #     let %a4 = (%a1, %a3);
    #     %a4
    # }
    class SGD(raf.Model):
        def build(self, model):
            self.model = model

        def reset(self):
            self.model.reset()

        @raf.model.trace
        def forward(self, dy):
            out = self.model()
            # grad of model, which will be replaced by AutoDiff
            dx = raf.relu_dx(self.model.x, out, dy)
            # update params
            new_x = raf.subtract(self.model.x, dx, out=self.model.x)
            trace_mutate_attr(self.model, "x", new_x)
            return out

    shape = [2, 3, 4]
    param = raf.array(np.random.randn(*shape), device=device)
    dy = raf.array(np.random.randn(*shape), device=device)
    model = Model(shape)
    sgd = SGD(model)
    # Interpreter
    model.x.update(param)
    out_1 = sgd(dy)
    new_x_1 = model.x
    # VM
    model.x.update(param)
    out = run_vm_model(sgd, device, [dy])
    out_2 = out[0]
    new_x_2 = out[1]
    # check inplace
    check(model.x, new_x_2)
    # check forward
    check(out_1, out_2)
    # check backward
    check(new_x_1, new_x_2)
    # check bytecode
    # # reg file size = 14
    # # instruction count = 16
    # opcode, fields # inst(text):
    # 0: 3 0 2   # load_const $2 Const[0]
    # 1: 20 2 64 2 64 1 1 0 3   # alloc_storage $3 $2 64 float64
    # 2: 21 3 0 2 64 1 3 4 2 3 4   # alloc_tensor $4 $3 $0 [2, 3, 4] float64
    # 3: 3 1 5   # load_const $5 Const[1]
    # 4: 23 1 6 1   # alloc_tuple $6 [$1]
    # 5: 23 1 7 4   # alloc_tuple $7 [$4]
    # 6: 33 5 2 1 1 4   # invoke_jit $5 (in: $1, out: $4)
    # 7: 3 2 8   # load_const $8 Const[2]
    # 8: 23 3 9 1 4 0   # alloc_tuple $9 [$1,$4,$0]
    # 9: 23 1 10 1   # alloc_tuple $10 [$1]
    # 10: 33 8 4 1 1 4 0 1   # invoke_jit $8 (in: $1, $4, $0, out: $1)
    # 11: 23 2 11 4 1   # alloc_tuple $11 [$4,$1]
    # 12: 1 11   # ret $11
    model.x.update(param)
    bytecode = compile_vm_model(sgd, device, [dy, model.x])
    assert bytecode.count("alloc_tensor") == 1


def test_chain():
    shape = (10, 20)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            a_1 = raf.add(x, x)
            a_prev = a_1
            for _ in range(5):
                a_i = raf.add(a_prev, x, out=a_prev)
                a_prev = a_i
            return a_i

    # fn (%x: Tensor[(10, 20), float64]) {
    #   let %x1 = raf.op.add(%x, %x, nullptr, nullptr);
    #   let %x2(share: %x1) = raf.op.add(%x1, %x, %x1, nullptr);
    #   let %x3(share: %x2) = raf.op.add(%x2, %x, %x2, nullptr);
    #   ...
    # }
    model = Model()
    device = "cpu"
    x = raf.array(np.random.randn(*shape), device=device)
    func = lower(model, [x])["main"]
    variables = extract_vars(func.body)
    prev_var = None
    for variable in variables:
        extended_var = ExtendedVar(variable)
        if prev_var is not None:
            assert extended_var.may_share == prev_var
        prev_var = extended_var

    bytecode = compile_vm_model(model, device, [x])
    assert bytecode.count("alloc_tensor") == 1


def test_reduce():
    shape = (4, 4)
    device = "cpu"

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.reduce(x, 0)

    model = Model()
    x, _ = randn(shape, device=device)
    bytecode = compile_vm_model(model, device, [x])
    assert bytecode.count("alloc_tensor") == 0


def test_reduce_with_list():
    shape = (4, 4)
    device = "cpu"

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            a = raf.add(x, y)
            b = raf.subtract(x, y)
            out = raf.reduce((a, b), 0)
            return out[0], out[1]

    model = Model()
    x, _ = randn(shape, device=device)
    y, _ = randn(shape, device=device)
    bytecode = compile_vm_model(model, device, [x, y])
    assert bytecode.count("alloc_tensor") == 2


def test_allreduce():
    shape = (4, 4)
    device = "cpu"

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.allreduce(x)

    model = Model()
    x, _ = randn(shape, device=device)
    bytecode = compile_vm_model(model, device, [x])
    assert bytecode.count("alloc_tensor") == 0


def test_allreduce_with_list():
    shape = (4, 4)
    device = "cpu"

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            a = raf.add(x, y)
            b = raf.subtract(x, y)
            out = raf.allreduce((a, b))
            return out[0], out[1]

    model = Model()
    x, _ = randn(shape, device=device)
    y, _ = randn(shape, device=device)
    bytecode = compile_vm_model(model, device, [x, y])
    assert bytecode.count("alloc_tensor") == 2


def test_simplify():
    def get_mod():
        add_op = raf._ffi.op.GetOp("raf.op.add")
        relu_op = raf._ffi.op.GetOp("raf.op.relu")
        zero = raf.ir.const(0.0, dtype="float32")
        null = raf.ir.const(None)

        data = raf.ir.var("x", shape=(16, 16))
        sb = ScopeBuilder()
        out = sb.let("a1", relay.Call(relu_op, [data]))
        out = sb.let("a2", relay.Call(add_op, [out, zero, data, null]))
        sb.ret(out)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    def expected():
        relu_op = raf._ffi.op.GetOp("raf.op.relu")

        data = raf.ir.var("x", shape=(16, 16))
        sb = ScopeBuilder()
        out = sb.let("a1", relay.Call(relu_op, [data]), data)
        sb.ret(out)
        func = relay.Function([data], sb.get())
        return pass_.InferType()(tvm.IRModule.from_expr(func))

    mod = get_mod()
    mod = optimize(mod)
    assert tvm.ir.structural_equal(mod["main"], expected()["main"])


if __name__ == "__main__":
    pytest.main([__file__])
