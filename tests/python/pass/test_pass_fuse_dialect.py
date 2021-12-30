# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import pytest
import mnm
from mnm.testing import run_infer_type, randn
import tvm
from tvm import relay


def optimize(mod):
    with mnm.device("cuda"):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseDialect()(mod)
        mod = mnm._ffi.pass_.InferType()(mod)
    return mod


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("matmul", ["matmul", "dense", "batch_matmul_tt"])
@pytest.mark.parametrize("act", [None, "relu", "gelu"])
@pytest.mark.parametrize("scaled_bias", [False, True])
def test_matmul_fusion(matmul, act, scaled_bias):
    class Model(mnm.Model):
        def build(self, act_op=None, beta=None):
            self.act_op = act_op
            self.beta = beta

        @mnm.model.trace
        def forward(self, x, w, bias):
            matmul_op = getattr(mnm._op.sym, matmul)
            out = matmul_op(x, w)
            if self.beta:
                out = mnm.add(out, mnm.multiply(bias, self.beta))
            else:
                out = mnm.add(out, bias)
            if self.act_op:
                out = self.act_op(out)
            return out

    def expected():
        add_op = mnm._ffi.op.GetOp("mnm.op.cutlass.add")
        multiply_op = mnm._ffi.op.GetOp("mnm.op.cutlass.multiply")
        matmul_op = mnm._ffi.op.GetOp("mnm.op.cutlass." + matmul)
        null = mnm.ir.const(None)

        x = mnm.ir.var("p", shape=xshape)
        w = mnm.ir.var("p1", shape=wshape)
        bias = mnm.ir.var("p2", shape=(30,))
        beta = mnm.ir.var("p3", shape=(1,))
        p4 = mnm.ir.var("p", relay.TupleType(()))
        p5 = mnm.ir.var("p", relay.TupleType(()))
        y = relay.Call(matmul_op, [x, w])
        params = [x, w, bias]
        if scaled_bias:
            bias = relay.Call(multiply_op, [bias, beta])
            params.append(beta)
        params.append(p4)
        params.append(p5)
        y = relay.Call(add_op, [y, bias, p4, p5])
        if act:
            act_op = mnm._ffi.op.GetOp("mnm.op.cutlass." + act)
            y = relay.Call(act_op, [y])
        f = relay.Function(params, y)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "cutlass")
        pattern_name = "batch_matmul_fusion" if matmul == "batch_matmul_tt" else "matmul_fusion"
        f = f.with_attr("PatternName", pattern_name)

        x = mnm.ir.var("x", shape=xshape)
        w = mnm.ir.var("w", shape=wshape)
        bias = mnm.ir.var("bias", shape=(30,))
        beta = mnm.ir.var("beta", shape=(1,))
        args = [x, w, bias]
        if scaled_bias:
            args.append(beta)
        body = relay.Call(f, args + [null, null])
        return relay.Function(args, body)

    if matmul == "matmul":
        xshape = (10, 20)
        wshape = (20, 30)
    elif matmul == "dense":
        xshape = (10, 20)
        wshape = (30, 20)
    else:  # "batch_matmul_tt"
        xshape = (4, 20, 10)
        wshape = (4, 30, 20)
    beta, _ = randn((1,), device="cpu")
    m_x, _ = randn(xshape, device="cpu")
    m_w, _ = randn(wshape, device="cpu")
    m_bias, _ = randn((30,), device="cpu")
    act_op = getattr(mnm._op.sym, act) if act is not None else act
    if scaled_bias:
        model = Model(act_op, beta)
    else:
        model = Model(act_op)
    mod = model._internal(m_x, m_w, m_bias).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_matmul_alone():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x, w):
            out = mnm.dense(x, w)
            out = mnm.multiply(out, self.c)
            return out

    def expected():
        multiply_op = mnm._ffi.op.GetOp("mnm.op.multiply")
        dense_op = mnm._ffi.op.GetOp("mnm.op.cublas.dense")

        x = mnm.ir.var("x", shape=(10, 20))
        w = mnm.ir.var("w", shape=(30, 20))
        c = mnm.ir.var("c", shape=(1,))
        y = relay.Call(dense_op, [x, w])
        y = relay.Call(multiply_op, [y, c])
        return relay.Function([x, w, c], y)

    m_x, _ = randn((10, 20), device="cpu")
    m_w, _ = randn((30, 20), device="cpu")
    model = Model()
    mod = model._internal(m_x, m_w).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
def test_conv2d_relu_fail():
    # do not fuse conv2d with channel mode NCHW
    device, dtype = "cuda", "float32"

    class Conv2D(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w):
            y = mnm.conv2d(x, w, stride=3, padding=1, dilation=1, groups=1)
            y = mnm.relu(y)
            return y

    xshape, wshape = (4, 256, 32, 32), (64, 256, 1, 1)
    m_x, _ = randn(xshape, device=device, dtype=dtype)
    m_w, _ = randn(wshape, device=device, dtype=dtype)
    model = Conv2D()
    mod = model._internal(m_x, m_w).mod
    mod = optimize(mod)
    assert mnm.ir.AsText(mod).count("cutlass") == 0


if __name__ == "__main__":
    pytest.main([__file__])
