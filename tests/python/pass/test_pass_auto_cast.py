# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, no-self-use, unused-argument
# pylint: disable=too-many-locals, too-many-arguments, attribute-defined-outside-init
import pytest
import torch

import raf
import tvm
from tvm import relay
from tvm.ir import PrimType
from raf.ir import AsText, ScopeBuilder
from raf.frontend.model import FrameworkModel
from raf.testing import randn, randn_torch, run_vm_model, check, get_testable_devices


def verify_correctness(model, device, args, ref_outs=None, tol=1e-5):
    # A helper function to verify the correctness
    args = [arg.to(device=device) for arg in args]
    model.to(device=device)
    ref_outs = model(*args) if ref_outs is None else ref_outs
    ref_outs = ref_outs if isinstance(ref_outs, (tuple, list)) else (ref_outs,)

    amp_model = raf.amp.autocast(model, args)
    outs = run_vm_model(amp_model, device, args)

    outs = outs if isinstance(outs, (tuple, list, raf._core.value.TupleValue)) else (outs,)
    assert len(ref_outs) == len(outs)
    for ref_out, out in zip(ref_outs, outs):
        check(ref_out, out, rtol=tol, atol=tol)


def verify_cast_num(model, args, expected):
    amp_model = raf.amp.autocast(model, args)
    mod = amp_model._internal(*args).mod
    text = AsText(raf._ffi.pass_.InferType()(mod)["main"])
    cast_cnt = 0
    for line in text.split("\n"):
        if line.find("raf.op.cast") != -1:
            cast_cnt += 1

    assert cast_cnt == expected, "Unexpected #cast: %d vs. %d\n%s" % (cast_cnt, expected, text)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_basic():
    device = "cuda"
    xshape = (1, 3, 224, 224)
    wshape = (32, 3, 3, 3)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w, b):
            x1 = raf.conv2d(x, w)
            x2 = raf.bias_add(x1, b)
            return x2

    model = Model()
    model.infer_mode()
    m_x, _ = randn(xshape, requires_grad=False)
    m_w, _ = randn(wshape, requires_grad=True)
    m_b, _ = randn((wshape[0],), requires_grad=True)
    args = [m_x, m_w, m_b]

    # cast x, w, b to fp16.
    verify_cast_num(model, args, 3)
    verify_correctness(model, device, args, tol=1e-1)

    def never_cast_bias_add(args, ret_type, amp_dtype):
        return [PrimType("float32"), PrimType("float32"), PrimType(None)]

    with raf.amp.CustomTypeHint({"raf.op.bias_add": never_cast_bias_add}):
        # cast x, w to fp16; cast x1 back to fp32.
        verify_cast_num(model, args, 3)
        verify_correctness(model, device, args, tol=1e-1)


@pytest.mark.parametrize("out_dtype", ["float16", "float32"])
def test_tuple_n_output_dtype(out_dtype):
    xshape = (1, 3, 224, 224)
    wshape = (32, 3, 3, 3)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            y = raf.conv2d(x, w)
            z = raf.relu(y)
            return (y, z)

    model = Model()
    m_x, _ = randn(xshape, requires_grad=False)
    m_w, _ = randn(wshape, requires_grad=True)
    args = [m_x, m_w]

    with raf.ir.PassContext(config={"raf.amp.out_dtype": out_dtype}):
        # Cast 2 inputs for both cases.
        # If output dtype is fp32, then cast 2 outputs back to fp32: Total 4 casts.
        # If output dtype if fp16, then do nothing: Total 2 casts.
        verify_cast_num(model, args, 4 if out_dtype == "float32" else 2)


def test_tuple_from_op():
    xshape = (2, 3, 224, 224)
    wshape = (32, 3, 3, 3)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            y = raf.conv2d(x, w)
            y = raf.split(y, 2)
            z = raf.concatenate(y)
            return z

    model = Model()
    m_x, _ = randn(xshape, requires_grad=False)
    m_w, _ = randn(wshape, requires_grad=True)
    args = [m_x, m_w]

    with raf.ir.PassContext(config={"raf.amp.out_dtype": "float16"}):
        # Cast 2 inputs.
        verify_cast_num(model, args, 2)


@pytest.mark.parametrize("out_dtype", ["float16", "float32"])
def test_existing_cast_with_always_op(out_dtype):
    xshape = (1, 3, 224, 224)
    wshape = (32, 3, 3, 3)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            fp32_x = raf.cast(x, "float32")
            fp32_w = raf.cast(w, "float32")
            fp32_out = raf.conv2d(fp32_x, fp32_w)
            return raf.cast(fp32_out, "float16")

    model = Model()
    m_x, _ = randn(xshape, requires_grad=False, dtype="float16")
    m_w, _ = randn(wshape, requires_grad=True, dtype="float16")
    args = [m_x, m_w]

    with raf.ir.PassContext(config={"raf.amp.out_dtype": out_dtype}):
        verify_cast_num(model, args, 0 if out_dtype == "float16" else 1)


@pytest.mark.parametrize("out_dtype", ["float16", "float32"])
def test_existing_cast_with_infer_op(out_dtype):
    xshape = (10, 10)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            fp32_x = raf.cast(x, "float32")
            fp32_out = raf.relu(fp32_x)
            return raf.cast(fp32_out, "float16")

    model = Model()
    m_x, _ = randn(xshape, requires_grad=False, dtype="float16")
    args = [m_x]

    with raf.ir.PassContext(config={"raf.amp.out_dtype": out_dtype}):
        verify_cast_num(model, args, 0 if out_dtype == "float16" else 1)


def test_inplace():
    xshape = (10, 10)
    wshape = (10, 10)

    def get_model():
        """matmul always produces fp16 output, so it is illegal for the new a1 (fp16)
        to share with data_x (fp32).
        """
        matmul_op = raf._ffi.op.GetOp("raf.op.matmul")

        data_x = raf.ir.var("x", shape=xshape, dtype="float32")
        data_w = raf.ir.var("w", shape=wshape, dtype="float32")

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(matmul_op, [data_x, data_w]))
        a_2 = sb.let("a2", a_1, may_share=data_x)
        sb.ret(a_2)
        func = relay.Function([data_x, data_w], sb.get())
        mod = tvm.IRModule.from_expr(func)
        return FrameworkModel(mod, mod, {}, {})

    model = get_model()
    m_x, _ = randn(xshape, dtype="float32")
    m_w, _ = randn(wshape, dtype="float32")
    args = [m_x, m_w]

    with raf.ir.PassContext(config={"raf.amp.out_dtype": "float16"}):
        verify_cast_num(model, args, 4)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_batch_norm_infer():
    shape = (8, 8, 8, 8)
    momentum = 0.1
    eps = 1e-3
    device = "cuda"

    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, device=device)
    m_m, t_m = randn_torch(stats_shape, device=device)
    m_v, t_v = randn_torch(stats_shape, device=device, positive=True)
    m_w, t_w = randn_torch(stats_shape, device=device)
    m_b, t_b = randn_torch(stats_shape, device=device)
    args = [m_x, m_m, m_v, m_w, m_b]

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):
            return raf.batch_norm_infer(m_x, m_m, m_v, m_w, m_b, momentum, eps)

    t_x_fp16 = t_x.to(torch.float16)
    t_y = torch.nn.functional.batch_norm(t_x_fp16, t_m, t_v, t_w, t_b, False, momentum, eps)

    model = TestModel()
    with raf.ir.PassContext(config={"raf.amp.out_dtype": "float16"}):
        amp_model = raf.amp.autocast(model, args)
        verify_correctness(amp_model, device, args, ref_outs=t_y)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_batch_norm_train():
    shape = (8, 8, 8, 8)
    momentum = 0.1
    eps = 1e-3
    device = "cuda"

    stats_shape = [shape[1]]
    m_x, t_x = randn_torch(shape, device=device, requires_grad=True)
    m_mean, t_mean = randn_torch(stats_shape, device=device)
    m_var, t_var = randn_torch(stats_shape, device=device, positive=True)
    m_w, t_w = randn_torch(stats_shape, device=device, requires_grad=True)
    m_b, t_b = randn_torch(stats_shape, device=device, requires_grad=True)
    args = [m_x, m_mean, m_var, m_w, m_b]

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_m, m_v, m_w, m_b):
            result = raf.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            return (result[0], result[1], result[2])

    t_x_fp16 = t_x.to(torch.float16)
    t_y = torch.nn.functional.batch_norm(t_x_fp16, t_mean, t_var, t_w, t_b, True, momentum, eps)

    model = TestModel()
    with raf.ir.PassContext(config={"raf.amp.out_dtype": "float16"}):
        amp_model = raf.amp.autocast(model, args)
        verify_correctness(amp_model, device, args, ref_outs=(t_y, t_mean, t_var))


def test_binary_ufunc():
    device = "cpu"
    shape = (10, 10)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, m_x, m_w, m_y):
            out = raf.matmul(m_x, m_w)
            new_out = raf.add(out, m_y, out=out)
            return new_out

    model = TestModel()
    m_x, _ = randn(shape, requires_grad=False)
    m_w, _ = randn(shape, requires_grad=False)
    m_y, _ = randn(shape, requires_grad=False)
    args = [m_x, m_w, m_y]
    verify_cast_num(model, args, 3)
    verify_correctness(model, device, args, tol=1e-2)


def test_cast_reuse():
    xshape = (32, 1024)
    wshape = (1024, 1000)

    class Model(raf.Model):
        def build(self):
            self.w, _ = randn(wshape, requires_grad=True)

        @raf.model.trace
        def forward(self, x):
            out1 = raf.matmul(x, self.w)  # Always cast.
            out6 = raf.matmul(x, self.w)  # Should reuse the cast ops of two inputs.
            out2 = raf.softmax(out1)  # Use a never cast op to produce a cast op.
            out3 = raf.exp(out1)  # This is a fuable op so it cannot reuse the cast op.
            out4 = raf.add(out2, out3)
            out5 = raf.add(out4, out6)
            return out5

    model = Model()
    model.infer_mode()
    m_x, _ = randn(xshape, requires_grad=False)
    verify_cast_num(model, [m_x], 5)


@pytest.mark.parametrize(
    "params",
    [
        # Majority is float16, cast 2 float32 to float16.
        (2, 3, 2),
        # Majority is float32, cast 2 float16 to float32.
        (3, 2, 2),
        # Total input > 5, force to cast all inputs to float32.
        (3, 4, 4),
    ],
)
def test_concatenate(params):
    shape = (12, 10)
    n_fp32_inputs, n_fp16_inputs, expected_cast_num = params

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, *args):
            return raf.concatenate(args, axis=0)

    model = Model()
    args = [randn(shape, requires_grad=False, dtype="float32")[0] for _ in range(n_fp32_inputs)] + [
        randn(shape, requires_grad=False, dtype="float16")[0] for _ in range(n_fp16_inputs)
    ]
    verify_cast_num(model, args, expected_cast_num)


@pytest.mark.parametrize("device", get_testable_devices())
def test_mean_dx(device):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf._op.sym.mean_dx(x, ())

    m_x, _ = randn((), dtype="float32", device=device)
    model = Model()
    verify_correctness(model, device, [m_x])


if __name__ == "__main__":
    pytest.main([__file__])
