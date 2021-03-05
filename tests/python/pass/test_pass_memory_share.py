# pylint: disable=protected-access
import pprint
import pytest
import numpy as np
import mnm
from mnm._op import sym
from mnm.model.trace import trace_mutate_attr
from mnm._ffi.pass_ import MemShare, InferType
from mnm._core.module import Module
from mnm._core.ir_ext import ExtendedVar, extended_var
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


def optimize(func):
    mod = Module.from_expr(func)
    mod = InferType(mod)
    mod = MemShare(mod)
    return mod["main"]


def lower(model, *data):
    func = model._internal(*data).mod['main']
    return optimize(func)


def checkir(variables, alias):
    for var in alias:
        assert var in variables
    for var in variables:
        assert isinstance(var, _relay.Var)
        var = ExtendedVar(var)
        may_share = var.may_share
        if var not in alias:
            assert may_share is None
        else:
            assert may_share == alias[var]


class BatchNorm(mnm.Model):  # pylint: disable=too-many-instance-attributes
    # pylint: disable=attribute-defined-outside-init
    def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_var = self.running_mean = None
        if affine:
            self.w = self.b = None
        self.reset()

    def reset(self):
        n_f = self.num_features
        self.running_mean = mnm.array(np.zeros(n_f, dtype="float32"),
                                      name="running_mean")
        self.running_var = mnm.array(np.ones(n_f, dtype="float32"),
                                     name="running_var")
        if self.affine:
            self.w = mnm.array(np.ones(n_f, dtype="float32"),
                               name="w")
            self.b = mnm.array(np.zeros(n_f, dtype="float32"),
                               name="b")

    @mnm.model.trace
    def forward(self, x):
        ret = sym.batch_norm_train(x=x,
                                   w=self.w,
                                   b=self.b,
                                   running_mean=self.running_mean,
                                   running_var=self.running_var,
                                   eps=self.eps,
                                   momentum=self.momentum)
        trace_mutate_attr(self, "running_mean", ret[1])
        trace_mutate_attr(self, "running_var", ret[2])
        return ret[0]


    @mnm.model.trace
    def forward_infer(self, x):
        ret = sym.batch_norm_infer(x=x,
                                   w=self.w,
                                   b=self.b,
                                   running_mean=self.running_mean,
                                   running_var=self.running_var,
                                   eps=self.eps,
                                   momentum=self.momentum)
        return ret


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
    func = lower(model, data)
    variables, _, _ = explicit_let_list(func.body)
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
    func = lower(model, data)
    variables, _, _ = explicit_let_list(func.body)
    running_mean = func.params[2]
    running_var = func.params[3]
    alias = {
        variables[3]: running_mean,
        variables[4]: running_var,
        variables[7]: running_mean,
        variables[8]: running_var,
    }
    checkir(variables, alias)


def test_add():
    shape = (2, 2)
    dtype = 'float32'
    data = mnm.array(np.ones(shape), dtype=dtype)

    class TestSuccess(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, shape):
            self.param = mnm.array(np.ones(shape), dtype=dtype)

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(self.param, x)
            trace_mutate_attr(self, "param", y)
            return y

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 2), float32], %param: Tensor[(2, 2), float32]) -> (Tensor[(2, 2), float32], Tensor[(2, 2), float32]) {
    #   let %a1 = mnm.op.add(%param, %x, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=Tensor[(2, 2), float32] */;
    #   let %a2 = (%a1, %a1);
    #   %a2
    # }
    # pylint: enable=line-too-long
    model = TestSuccess(shape=shape)
    func = lower(model, data)
    variables, _, _ = explicit_let_list(func.body)
    param = func.params[1]
    alias = {
        variables[0]: param
    }
    checkir(variables, alias)

    class TestFailure(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, shape):
            self.param = mnm.array(np.ones(shape), dtype=dtype)

        @mnm.model.trace
        def forward(self, x):
            origin = self.param
            y = mnm.add(origin, x)
            trace_mutate_attr(self, "param", y)
            return origin

    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 2), float32], %param: Tensor[(2, 2), float32]) -> (Tensor[(2, 2), float32], Tensor[(2, 2), float32]) {
    #     let %a1 = mnm.op.add(%param, %x, -114514 /* ty=int64 */, -114514 /* ty=int64 */) /* ty=Tensor[(2, 2), float32] */;
    #     let %a2 = (%param, %a1);
    #     %a2
    # }
    # pylint: enable=line-too-long
    model = TestFailure(shape=shape)
    func = lower(model, data)
    variables, _, _ = explicit_let_list(func.body)
    alias = {}
    checkir(variables, alias)


def test_cos():
    shape = (2, 2)
    dtype = 'float32'

    class Test(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, shape, dtype):
            self.param = mnm.array(np.ones(shape), dtype=dtype)

        @mnm.model.trace
        def forward(self):
            x = mnm.cos(self.param)
            trace_mutate_attr(self, "param", x)
            x = mnm.cos(self.param)
            trace_mutate_attr(self, "param", x)
            origin = self.param
            x = mnm.cos(self.param)
            trace_mutate_attr(self, "param", x)  # cannot be inplaced
            return origin

    # pylint: disable=line-too-long
    # fn (%param: Tensor[(2, 2), float32]) -> (Tensor[(2, 2), float32], Tensor[(2, 2), float32]) {
    #   let %a1 = mnm.op.cos(%param) /* ty=Tensor[(2, 2), float32] */;
    #   let %a2 = mnm.op.cos(%a1) /* ty=Tensor[(2, 2), float32] */;
    #   let %a3 = mnm.op.cos(%a2) /* ty=Tensor[(2, 2), float32] */;
    #   let %a4 = (%a2, %a3);
    #   %a4
    # }
    # pylint: enable=line-too-long
    model = Test(shape=shape, dtype=dtype)
    func = lower(model)
    variables, _, _ = explicit_let_list(func.body)
    param = func.params[0]
    alias = {
        variables[0]: param,
        variables[1]: param,
    }
    checkir(variables, alias)


def test_if():
    # pylint: disable=too-many-locals
    # define var:
    cond = extended_var("cond", dtype="bool", shape=())
    x = extended_var("x", dtype="float32", shape=(2, 2))
    a_1 = extended_var("a1", may_share=x)
    a_2 = extended_var("a2")
    x_1_t = extended_var("x1_t", may_share=x)
    x_1_f = extended_var("x1_f", may_share=x)
    a_3 = extended_var("a3", may_share=x)
    cos = _relay.op.get("mnm.op.cos")
    log = _relay.op.get("mnm.op.log")
    add = _relay.op.get("mnm.op.add")
    # func1:
    # pylint: disable=line-too-long
    # fn (%cond: bool, %x: Tensor[(2, 2), float32]) -> Tensor[(2, 2), float32] {
    #   let %a1 = mnm.op.cos(%x) /* ty=Tensor[(2, 2), float32] */;
    #   let %a2 = if (%cond) {
    #     let %x1_t = mnm.op.cos(%a1) /* ty=Tensor[(2, 2), float32] */;
    #     %x1_t
    #   } else {
    #     let %x1_f = mnm.op.log(%a1) /* ty=Tensor[(2, 2), float32] */;
    #     %x1_f
    #   };
    #   let %a3 = mnm.op.cos(%a2) /* ty=Tensor[(2, 2), float32] */;
    #   %a3
    # }
    # pylint: enable=line-too-long
    body_t = _relay.Let(x_1_t, _relay.Call(cos, [a_1]), x_1_t)
    body_f = _relay.Let(x_1_f, _relay.Call(log, [a_1]), x_1_f)
    body = _relay.Let(a_3, _relay.Call(cos, [a_2]), a_3)
    body = _relay.Let(a_2, _relay.If(cond, body_t, body_f), body)
    body = _relay.Let(a_1, _relay.Call(cos, [x]), body)
    func = _relay.Function([cond, x], body)
    # check ir
    func = optimize(func)
    variables, _, _ = explicit_let_list(func.body)
    alias = {
        a_1: x,
        x_1_t: x,
        x_1_f: x,
        a_3: x
    }
    checkir(variables, alias)
    # func2:
    # pylint: disable=line-too-long
    # fn (%cond: bool, %x: Tensor[(2, 2), float32]) {
    #   let %a1 = mnm.op.cos(%x);
    #   let %a2 = if (%cond) {
    #     let %x1_t = mnm.op.cos(%a1);
    #     %x1_t
    #   } else {
    #     let %x1_f = mnm.op.log(%a1);
    #     %x1_f
    #   };
    #   let %a3 = mnm.op.add(%a1, %a2);
    #   %a3
    # }
    # pylint: enable=line-too-long
    body_t = _relay.Let(x_1_t, _relay.Call(cos, [a_1]), x_1_t)
    body_f = _relay.Let(x_1_f, _relay.Call(log, [a_1]), x_1_f)
    body = _relay.Let(a_3, _relay.Call(add, [a_1, a_2]), a_3)
    body = _relay.Let(a_2, _relay.If(cond, body_t, body_f), body)
    body = _relay.Let(a_1, _relay.Call(cos, [x]), body)
    func = _relay.Function([cond, x], body)
    # check ir
    func = optimize(func)
    variables, _, _ = explicit_let_list(func.body)
    alias = {
        a_1: x,
        a_3: x
    }
    checkir(variables, alias)


if __name__ == "__main__":
    pytest.main([__file__])
