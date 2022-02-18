# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
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


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
def test_duplicate():
    class Model(mnm.Model):
        def build(self):
            self.const1, _ = randn((1,), device="cpu")
            self.const2, _ = randn((1,), device="cpu")

        @mnm.model.trace
        def forward(self, data, weight1, weight2):
            out = mnm.dense(data, weight1)
            out = mnm.add(out, self.const1)
            out = mnm.dense(out, weight2)
            out = mnm.add(out, self.const2)
            return out

    def expected():
        dense_op = mnm._ffi.op.GetOp("mnm.op.cutlass.dense")
        add_op = mnm._ffi.op.GetOp("mnm.op.cutlass.add")
        null = mnm.ir.const(None)

        p_0 = mnm.ir.var("p", shape=(10, 10))
        p_1 = mnm.ir.var("p1", shape=(10, 10))
        p_2 = mnm.ir.var("p2", shape=(1,))
        p_3 = mnm.ir.var("p3", relay.TupleType(()))
        p_4 = mnm.ir.var("p4", relay.TupleType(()))

        # Fused function (should only have one used by both calls)
        out = relay.Call(dense_op, [p_0, p_1])
        out = relay.Call(add_op, [out, p_2, p_3, p_4])
        func = relay.Function([p_0, p_1, p_2, p_3, p_4], out)
        func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Dialect", "cutlass")
        func = func.with_attr("PatternName", "matmul_fusion")

        # Main function
        data = mnm.ir.var("data", shape=(10, 10))
        weight1 = mnm.ir.var("weight1", shape=(10, 10))
        weight2 = mnm.ir.var("weight2", shape=(10, 10))
        const1 = mnm.ir.var("const1", shape=(1,))
        const2 = mnm.ir.var("const2", shape=(1,))

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
