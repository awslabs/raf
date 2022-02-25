# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import pytest
import raf
from raf.testing import run_infer_type, randn
import tvm
from tvm import relay


def optimize(mod):
    with raf.device("cuda"):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.InferType()(mod)
    return mod


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("matmul", ["matmul", "dense", "batch_matmul_tt"])
@pytest.mark.parametrize("act", [None, "relu", "gelu"])
@pytest.mark.parametrize("scaled_bias", [False, True])
def test_matmul_fusion(matmul, act, scaled_bias):
    class Model(raf.Model):
        def build(self, act_op=None, beta=None):
            self.act_op = act_op
            self.beta = beta

        @raf.model.trace
        def forward(self, x, w, bias):
            matmul_op = getattr(raf._op.sym, matmul)
            out = matmul_op(x, w)
            if self.beta:
                out = raf.add(out, raf.multiply(bias, self.beta))
            else:
                out = raf.add(out, bias)
            if self.act_op:
                out = self.act_op(out)
            return out

    def expected():
        add_op = raf._ffi.op.GetOp("raf.op.cutlass.add")
        multiply_op = raf._ffi.op.GetOp("raf.op.cutlass.multiply")
        matmul_op = raf._ffi.op.GetOp("raf.op.cutlass." + matmul)
        null = raf.ir.const(None)

        x = raf.ir.var("p", shape=xshape)
        w = raf.ir.var("p1", shape=wshape)
        bias = raf.ir.var("p2", shape=(30,))
        beta = raf.ir.var("p3", shape=(1,))
        p4 = raf.ir.var("p", relay.TupleType(()))
        p5 = raf.ir.var("p", relay.TupleType(()))
        y = relay.Call(matmul_op, [x, w])
        params = [x, w, bias]
        if scaled_bias:
            bias = relay.Call(multiply_op, [bias, beta])
            params.append(beta)
        params.append(p4)
        params.append(p5)
        y = relay.Call(add_op, [y, bias, p4, p5])
        if act:
            act_op = raf._ffi.op.GetOp("raf.op.cutlass." + act)
            y = relay.Call(act_op, [y])
        f = relay.Function(params, y)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "cutlass")
        pattern_name = "batch_matmul_fusion" if matmul == "batch_matmul_tt" else "matmul_fusion"
        f = f.with_attr("PatternName", pattern_name)

        x = raf.ir.var("x", shape=xshape)
        w = raf.ir.var("w", shape=wshape)
        bias = raf.ir.var("bias", shape=(30,))
        beta = raf.ir.var("beta", shape=(1,))
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
    act_op = getattr(raf._op.sym, act) if act is not None else act
    if scaled_bias:
        model = Model(act_op, beta)
    else:
        model = Model(act_op)
    mod = model._internal(m_x, m_w, m_bias).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_matmul_alone():
    konst, _ = randn((1,), device="cpu")

    class Model(raf.Model):
        def build(self):
            self.c = konst

        @raf.model.trace
        def forward(self, x, w):
            out = raf.dense(x, w)
            out = raf.multiply(out, self.c)
            return out

    def expected():
        multiply_op = raf._ffi.op.GetOp("raf.op.multiply")
        dense_op = raf._ffi.op.GetOp("raf.op.cublas.dense")

        x = raf.ir.var("x", shape=(10, 20))
        w = raf.ir.var("w", shape=(30, 20))
        c = raf.ir.var("c", shape=(1,))
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


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
def test_conv2d_relu_fail():
    # do not fuse conv2d with channel mode NCHW
    device, dtype = "cuda", "float32"

    class Conv2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            y = raf.conv2d(x, w, stride=3, padding=1, dilation=1, groups=1)
            y = raf.relu(y)
            return y

    xshape, wshape = (4, 256, 32, 32), (64, 256, 1, 1)
    m_x, _ = randn(xshape, device=device, dtype=dtype)
    m_w, _ = randn(wshape, device=device, dtype=dtype)
    model = Conv2D()
    mod = model._internal(m_x, m_w).mod
    mod = optimize(mod)
    assert raf.ir.AsText(mod).count("cutlass") == 0


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
def test_duplicate():
    class Model(raf.Model):
        def build(self):
            self.const1, _ = randn((1,), device="cpu")
            self.const2, _ = randn((1,), device="cpu")

        @raf.model.trace
        def forward(self, data, weight1, weight2):
            out = raf.dense(data, weight1)
            out = raf.add(out, self.const1)
            out = raf.dense(out, weight2)
            out = raf.add(out, self.const2)
            return out

    def expected():
        dense_op = raf._ffi.op.GetOp("raf.op.cutlass.dense")
        add_op = raf._ffi.op.GetOp("raf.op.cutlass.add")
        null = raf.ir.const(None)

        p_0 = raf.ir.var("p", shape=(10, 10))
        p_1 = raf.ir.var("p1", shape=(10, 10))
        p_2 = raf.ir.var("p2", shape=(1,))
        p_3 = raf.ir.var("p3", relay.TupleType(()))
        p_4 = raf.ir.var("p4", relay.TupleType(()))

        # Fused function (should only have one used by both calls)
        out = relay.Call(dense_op, [p_0, p_1])
        out = relay.Call(add_op, [out, p_2, p_3, p_4])
        func = relay.Function([p_0, p_1, p_2, p_3, p_4], out)
        func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Dialect", "cutlass")
        func = func.with_attr("PatternName", "matmul_fusion")

        # Main function
        data = raf.ir.var("data", shape=(10, 10))
        weight1 = raf.ir.var("weight1", shape=(10, 10))
        weight2 = raf.ir.var("weight2", shape=(10, 10))
        const1 = raf.ir.var("const1", shape=(1,))
        const2 = raf.ir.var("const2", shape=(1,))

        out = relay.Call(func, [data, weight1, const1, null, null])
        out = relay.Call(func, [out, weight2, const2, null, null])
        return relay.Function([data, weight1, weight2, const1, const2], out)

    m_x, _ = randn((10, 10), device="cpu")
    m_w1, _ = randn((10, 10), device="cpu")
    m_w2, _ = randn((10, 10), device="cpu")
    model = Model()
    mod = model._internal(m_x, m_w1, m_w2).mod
    mod = optimize(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
