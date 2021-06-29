# pylint: disable=protected-access, no-self-use, unused-argument
import pytest
import mnm
from tvm.ir import PrimType
from mnm.ir import AsText
from mnm.testing import randn, check


def verify_correctness(model, device, args):
    # A helper function to verify the correctness
    args = [arg.to(device=device) for arg in args]
    model.to(device=device)
    ref_outs = model(*args)
    ref_outs = ref_outs if isinstance(ref_outs, (tuple, list)) else (ref_outs,)

    amp_model = mnm.amp.autocast(model, args)
    outs = amp_model(*args)

    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    assert len(ref_outs) == len(outs)
    for ref_out, out in zip(ref_outs, outs):
        check(ref_out, out, rtol=0.1, atol=0.1)


def verify_cast_num(model, args, expected):
    amp_model = mnm.amp.autocast(model, args)
    mod = amp_model._internal(*args).mod
    text = AsText(mod["main"])
    cast_cnt = 0
    for line in text.split("\n"):
        if line.find("mnm.op.cast") != -1:
            cast_cnt += 1

    assert cast_cnt == expected, "Unexpected #cast: %d vs. %d\n%s" % (cast_cnt, expected, text)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_basic():
    device = "cuda"
    xshape = (1, 3, 224, 224)
    wshape = (32, 3, 3, 3)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w, b):
            x1 = mnm.conv2d(x, w)
            x2 = mnm.bias_add(x1, b)
            return x2

    model = Model()
    model.infer_mode()
    m_x, _ = randn(xshape, requires_grad=False)
    m_w, _ = randn(wshape, requires_grad=True)
    m_b, _ = randn((wshape[0],), requires_grad=True)
    args = [m_x, m_w, m_b]

    # cast x, w, b to fp16; cast x2 back to fp32.
    verify_cast_num(model, args, 3)
    verify_correctness(model, device, args)

    def disable_bias_add(args, ret_type):
        return [PrimType("float32"), PrimType("float32"), PrimType(None), PrimType("float32")]

    with mnm.amp.CustomTypeHint({"mnm.op.bias_add": disable_bias_add}):
        # cast x, w to fp16; cast a1 back to fp32; cast a2 to fp16
        verify_cast_num(model, args, 4)
        verify_correctness(model, device, args)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("out_dtype", ["float16", "float32"])
def test_tuple_n_output_dtype(out_dtype):
    shape = (5, 4, 6, 9)
    device = "cuda"

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, dy, x, w, b):
            y = mnm.batch_norm_train_dxwb(dy, x, w, b, 1e-5)
            z = mnm.add(dy, x)
            return (y[0], z, y[1], y[2])

    model = Model()
    m_dy, _ = randn(shape, dtype="float32")
    m_x, _ = randn(shape, dtype="float32")
    m_w, _ = randn((shape[1],), dtype="float32")
    m_b, _ = randn((shape[1],), dtype="float32")
    args = [m_dy, m_x, m_w, m_b]

    with mnm.ir.PassContext(config={"mnm.amp.out_dtype": out_dtype}):
        # cast dy, x to fp16 = 2 casts.
        # If output dtype is fp32, then only cast y[0] to fp32: Total 3 casts.
        # If output dtype if fp16, then cast the rest 3 outputs to fp16: Total 5 casts.
        verify_cast_num(model, args, 3 if out_dtype == "float32" else 5)
        verify_correctness(model, device, args)


if __name__ == "__main__":
    pytest.main([__file__])
