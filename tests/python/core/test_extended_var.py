# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm.model.trace import trace_mutate_attr
from mnm._core.ndarray import ndarray
from mnm._op import sym
from mnm._core.core_utils import get_chained_attr
from mnm._core.ir_ext import ExtendedVar
from mnm._lib import tvm as _tvm


def explicit_let_list(body):
    if isinstance(body, _tvm.relay.Var):
        return ([], [], body)
    assert isinstance(body, _tvm.relay.Let)
    variables, exprs, ret = explicit_let_list(body.body)
    variables = [body.var] + variables
    exprs = [body.value] + exprs
    return variables, exprs, ret


def checkir(variables, may_share):
    for var in variables:
        var = ExtendedVar(var)
        if var not in may_share:
            assert var.may_share is None
        else:
            assert var.may_share == may_share[var]


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    n_x = n_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


# TODO(@hzfan): use BatchNorm in python/mnm/model/nn.py
class BatchNorm(mnm.Model):
    # pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
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
        self.running_mean = ndarray(np.zeros(n_f, dtype="float32"),
                                    name="running_mean",
                                    ctx=get_chained_attr(self, ["running_mean", "ctx"], "cpu"))
        self.running_var = ndarray(np.ones(n_f, dtype="float32"),
                                   name="running_var",
                                   ctx=get_chained_attr(self, ["running_var", "ctx"], "cpu"))
        if self.affine:
            self.w = ndarray(np.ones(n_f, dtype="float32"),
                             name="w", ctx=get_chained_attr(self, ["w", "ctx"], "cpu"))
            self.b = ndarray(np.zeros(n_f, dtype="float32"),
                             name="b", ctx=get_chained_attr(self, ["b", "ctx"], "cpu"))

    # pylint: enable=attribute-defined-outside-init

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
    # pylint: disable=line-too-long
    # fn (%x: Tensor[(2, 3, 4, 5), float32], %batch_norm.b: Tensor[(3), float32], %batch_norm.running_mean: Tensor[(3), float32], %batch_norm.running_var: Tensor[(3), float32], %batch_norm.w: Tensor[(3), float32]) {
    #     let %a1 = mnm.op.batch_norm_train(%x, %batch_norm.running_mean, %batch_norm.running_var, %batch_norm.w, %batch_norm.b, -114514, -114514);
    #     let %a2 = %a1.0;
    #     let %a3 = mnm.op.relu(%a2);
    #     let %a4 = %a1.1;
    #     let %a5 = %a1.2;
    #     let %a6 = (%a3, %a4, %a5);
    #     %a6
    # }
    class Test(mnm.Model):
        # pylint: disable=attribute-defined-outside-init
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            x = self.batch_norm(x)
            x = mnm.relu(x)
            return x

    shape = (2, 3, 4, 5)
    dtype = 'float32'
    data = mnm.array(np.ones(shape), dtype=dtype)
    model = Test(num_features=3)
    func = model._internal(data).func
    variables, _, _ = explicit_let_list(func.body)
    running_mean = func.params[2]
    running_var = func.params[3]
    may_share = {
        variables[3]: running_mean,
        variables[4]: running_var,
    }
    checkir(variables, may_share)


def test_grad():
    # pylint: disable=too-many-locals, too-many-arguments, attribute-defined-outside-init
    class Model(mnm.Model):
        def build(self, shape):
            self.shape = shape
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*self.shape))

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
    dy = mnm.array(np.random.randn(*shape))
    model = Model(shape)
    sgd = SGD(model)
    # IR with SGD
    model.x = mnm.array(param)
    out_1 = sgd(dy)
    new_x_1 = model.x
    # IR without SGD
    model.x = mnm.array(param)
    model.x.requires_grad = True
    out_2 = model()
    out_2.backward(dy)
    new_x_2 = mnm.subtract(model.x, model.x.grad)
    # check forward
    check(out_1, out_2)
    # check backward
    check(new_x_1, new_x_2)
    # check ir
    func = sgd._internal(dy).func
    variables, _, _ = explicit_let_list(func.body)
    modelx = func.params[1]
    may_share = {
        variables[2]: modelx,  # %a3 is an may_share of %model.x
    }
    checkir(variables, may_share)


if __name__ == "__main__":
    pytest.main([__file__])
