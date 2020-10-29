import pprint
import numpy as np
import pytest
import mnm
from mnm.model.trace import trace_mutate_attr
from mnm.testing import get_ctx_list, check, compile_vm_model, run_vm_model
from mnm._core.ir_ext import ExtendedVar
from mnm._lib import tvm as _tvm
from mnm._lib import relay as _relay


def explicit_let_list(body):
    if isinstance(body, _tvm.relay.Var):
        return ([], [], body)
    assert isinstance(body, _tvm.relay.Let)
    variables, exprs, ret = explicit_let_list(body.body)
    if isinstance(body.value, _relay.If):
        tvs, tes, _ = explicit_let_list(body.value.true_branch)
        fvs, fes, _ = explicit_let_list(body.value.false_branch)
        variables = tvs + fvs + variables
        exprs = tes + fes + variables
    variables = [body.var] + variables
    exprs = [body.value] + exprs
    return variables, exprs, ret


def debug_dump(expr):
    print(expr)
    variables, _, _ = explicit_let_list(expr.body)
    may_share = {v: ExtendedVar(v).may_share for v in variables}
    pprint.pprint(may_share)


@pytest.mark.parametrize("ctx", get_ctx_list())
def test_grad(ctx):
    # pylint: disable=too-many-locals, too-many-arguments, attribute-defined-outside-init
    class Model(mnm.Model):
        def build(self, shape):
            self.shape = shape
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*self.shape), ctx=ctx)

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
            new_x = mnm.subtract(self.model.x, dx)
            trace_mutate_attr(self.model, "x", new_x)
            return out

    shape = [2, 3, 4]
    param = np.random.randn(*shape)
    dy = mnm.array(np.random.randn(*shape), ctx=ctx)
    model = Model(shape)
    sgd = SGD(model)
    # Interpreter
    model.x = mnm.array(param, ctx=ctx)
    out_1 = sgd(dy)
    new_x_1 = model.x
    # VM
    model.x = mnm.array(param, ctx=ctx)
    out = run_vm_model(sgd, ctx, [dy, model.x])
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
    model.x = mnm.array(param, ctx=ctx)
    bytecode = compile_vm_model(sgd, ctx, [dy, model.x])
    assert bytecode.count("alloc_tensor") == 2


if __name__ == "__main__":
    pytest.main([__file__])
