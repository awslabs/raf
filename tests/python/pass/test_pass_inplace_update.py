# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm._ffi import pass_
from mnm._core.ir_ext import ExtendedVar
from mnm.model.nn import BatchNorm
from mnm.model.trace import trace_mutate_attr
from mnm.testing import get_device_list, compile_vm_model, run_vm_model, check
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


def lower(model, *data):
    mod = model._internal(*data).mod
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
    dtype = 'float32'
    data = mnm.array(np.ones(shape), dtype=dtype)

    class Test1(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = mnm.relu(x)
            return x

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) -> (Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) {
    #   let %a1 = mnm.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a2 = %a1.0;
    #   let %a3 = mnm.op.relu(%a2) /* ty=Tensor[(2, 3, 4, 5), float32] */;
    #   let %a4 = %a1.1;
    #   let %a5 = %a1.2;
    #   let %a6 = (%a3, %a4, %a5);
    #   %a6
    # }
    # pylint: enable=line-too-long
    model = Test1(num_features=3)
    func = lower(model, data)["main"]
    variables = extract_vars(func.body)
    running_mean = func.params[2]
    running_var = func.params[3]
    alias = {
        variables[3]: running_mean,
        variables[4]: running_var,
    }
    checkir(variables, alias)

    class Test2(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = mnm.relu(x)
            # typically batch_norm is not used in this way:
            # two batch_norm generally correspond to two BatchNorm models.
            # for test only here.
            x = self.batch_norm(x)
            return x

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) -> (Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) {
    #   let %a1 = mnm.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a2 = %a1.0;
    #   let %a3 = mnm.op.relu(%a2) /* ty=Tensor[(2, 3, 4, 5), float32] */;
    #   let %a4 = %a1.1;
    #   let %a5 = %a1.2;
    #   let %a6 = mnm.op.batch_norm_train(%a3, %a4, %a5, %batch_norm.w, %batch_norm.b, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=(Tensor[(2, 3, 4, 5), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
    #   let %a7 = %a6.0;
    #   let %a8 = %a6.1;
    #   let %a9 = %a6.2;
    #   let %a10 = (%a7, %a8, %a9);
    #   %a10
    # }
    # pylint: enable=line-too-long
    model = Test2(num_features=3)
    func = lower(model, data)["main"]
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


@pytest.mark.parametrize("device", get_device_list())
def test_grad(device):
    # pylint: disable=too-many-locals, too-many-arguments, attribute-defined-outside-init
    class Model(mnm.Model):
        def build(self, shape):
            self.shape = shape
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*self.shape), device=device)

        @mnm.model.trace
        def forward(self):  # pylint: disable=no-self-use
            return mnm.relu(self.x)

    # fn (%dy: Tensor[(2, 3, 4), float64], %model.x: Tensor[(2, 3, 4), float64]) {
    #     let %a1 = mnm.op.relu(%model.x);
    #     let %a2 = mnm.op.relu_dx(%model.x, %a1, %dy);
    #     let %a3 = mnm.op.subtract(%model.x, %a2, -114514, -114514);
    #     let %a4 = (%a1, %a3);
    #     %a4
    # }
    class SGD(mnm.Model):
        def build(self, model):
            self.model = model

        def reset(self):
            self.model.reset()

        @mnm.model.trace
        def forward(self, dy):
            out = self.model()
            # grad of model, which will be replaced by AutoDiff
            dx = mnm.relu_dx(self.model.x, out, dy)
            # update params
            new_x = mnm.subtract(self.model.x, dx, out=self.model.x)
            trace_mutate_attr(self.model, "x", new_x)
            return out

    shape = [2, 3, 4]
    param = mnm.array(np.random.randn(*shape), device=device)
    dy = mnm.array(np.random.randn(*shape), device=device)
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
    #  0: 3 0 2   # load_const $2 Const[0]
    #  1: 20 2 64 2 64 1 1 0 3   # alloc_storage $3 $2 64 float64
    #  2: 21 3 0 2 64 1 3 4 2 3 4   # alloc_tensor $4 $3 $0 [2, 3, 4] float64
    #  3: 3 1 5   # load_const $5 Const[1]
    #  4: 33 5 2 1 1 4   # invoke_jit $5 (in: $1, out: $4)
    #  5: 3 2 6   # load_const $6 Const[2]
    #  6: 20 6 64 2 64 1 1 0 7   # alloc_storage $7 $6 64 float64
    #  7: 21 7 0 2 64 1 3 8 2 3 4   # alloc_tensor $8 $7 $0 [2, 3, 4] float64
    #  8: 3 3 9   # load_const $9 Const[3]
    #  9: 33 9 4 1 1 4 0 8   # invoke_jit $9 (in: $1, $4, $0, out: $8)
    # 10: 3 4 10   # load_const $10 Const[4]
    # 11: 3 5 11   # load_const $11 Const[5]
    # 12: 3 6 12   # load_const $12 Const[6]
    # 13: 33 10 5 1 1 8 11 12 1   # invoke_jit $10 (in: $1, $8, $11, $12, out: $1)
    # 14: 23 2 13 4 1   # alloc_tuple $13 [$4,$1]
    # 15: 1 13   # ret $13
    model.x.update(param)
    bytecode = compile_vm_model(sgd, device, [dy, model.x])
    assert bytecode.count("alloc_tensor") == 2


if __name__ == "__main__":
    pytest.main([__file__])
