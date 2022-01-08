# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import numpy as np
import pytest
import mnm
from mnm.ir import ScopeBuilder
from mnm.model import Conv2d
from mnm.model.trace import trace_mutate_attr
from mnm.testing import run_infer_type, randn
import tvm
from tvm import relay


def fuse_module(mod, fuse_dialect=False):
    mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
    mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
    if fuse_dialect:
        mod = mnm._ffi.pass_.FuseDialect()(mod)
    mod = mnm._ffi.pass_.FuseTVM()(mod)
    mod = mnm._ffi.pass_.InferType()(mod)
    return mod


def test_fuse_simple():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.c)
            y = mnm.relu(y)
            y = mnm.log(y)
            return y

    def expected(shape):
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.tvm.relu")
        log_op = mnm._ffi.op.GetOp("mnm.op.tvm.log")
        null = mnm.ir.const(None)

        x = mnm.ir.var("p0", shape=shape)
        y = mnm.ir.var("p1", shape=(1,))
        p2 = mnm.ir.var("p2", relay.TupleType(()))
        p3 = mnm.ir.var("p3", relay.TupleType(()))
        z = relay.Call(add_op, [x, y, p2, p3])
        z = relay.Call(log_op, [relay.Call(relu_op, [z])])
        f1 = relay.Function([x, y, p2, p3], z)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f1 = f1.with_attr("Dialect", "tvm")
        x = mnm.ir.var("x", shape=shape)
        y = mnm.ir.var("c", shape=(1,))
        ret = relay.Call(f1, [x, y, null, null])
        return relay.Function([x, y], ret)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    mod = fuse_module(mod)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod["main"], func_expected)


def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)
            self.conv2 = Conv2d(16, 16, kernel_size=(1, 1), padding=0, bias=False)
            self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @mnm.model.trace
        def forward(self, x):
            x = mnm.add(x, self.c)
            y = self.conv1(x)
            # this is the next dominator.
            y1 = mnm.add(y, self.c)
            y = mnm.add(y, y1)
            # second path
            z2 = self.conv2(y)
            z3 = self.conv3(y)
            # add can only be fused to z1
            z = mnm.add(z2, z3)
            return z

    def expected():
        v_zero = mnm.ir.const([0])
        v_one = mnm.ir.const([1])
        konst1 = mnm.ir.const(1)
        konst_nchw = mnm.ir.const("NCHW")
        konst_oihw = mnm.ir.const("OIHW")
        null = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        tvm_add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        tvm_conv2d_op = mnm._ffi.op.GetOp("mnm.op.tvm.conv2d")

        # segment 1
        x = mnm.ir.var("p0", shape=(1, 16, 64, 64))
        w = mnm.ir.var("p1", shape=(16, 16, 3, 3))
        p2 = mnm.ir.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = mnm.ir.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = mnm.ir.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = mnm.ir.var("p5", "int64")
        p6 = mnm.ir.var("p6", "int64")
        p7 = mnm.ir.var("p7", "int64")
        p8 = mnm.ir.var("p8", "int64")
        p9 = mnm.ir.var("p9", shape=(1,))
        p10 = mnm.ir.var("p10", relay.TupleType(()))
        p11 = mnm.ir.var("p11", relay.TupleType(()))
        p12 = mnm.ir.var("p12", relay.TupleType(()))
        p13 = mnm.ir.var("p13", relay.TupleType(()))
        y = relay.Call(tvm_conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y1 = relay.Call(tvm_add_op, [y, p9, p10, p11])
        y = relay.Call(tvm_add_op, [y, y1, p12, p13])
        f1 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f1 = f1.with_attr("Dialect", "tvm")

        # segment 3
        x = mnm.ir.var("p0", shape=(1, 16, 64, 64))
        w = mnm.ir.var("p1", shape=(16, 16, 1, 1))
        p2 = mnm.ir.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = mnm.ir.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = mnm.ir.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = mnm.ir.var("p5", "int64")
        p6 = mnm.ir.var("p6", "int64")
        p7 = mnm.ir.var("p7", "int64")
        p8 = mnm.ir.var("p8", "int64")
        p9 = mnm.ir.var("p9", shape=(1, 16, 64, 64))
        p10 = mnm.ir.var("p10", relay.TupleType(()))
        p11 = mnm.ir.var("p11", relay.TupleType(()))
        y = relay.Call(tvm_conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y = relay.Call(tvm_add_op, [y, p9, p10, p11])
        f3 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11], y)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f3 = f3.with_attr("Dialect", "tvm")

        # compose
        x = mnm.ir.var("x", shape=(1, 16, 64, 64))
        c = mnm.ir.var("c", shape=(1,))
        w1 = mnm.ir.var("conv1.w", shape=(16, 16, 3, 3))
        w2 = mnm.ir.var("conv2.w", shape=(16, 16, 1, 1))
        w3 = mnm.ir.var("conv3.w", shape=(16, 16, 3, 3))
        y1 = relay.Call(add_op, [x, c, null, null])
        y2 = relay.Call(
            f1,
            [
                y1,
                w1,
                v_one,
                v_one,
                v_one,
                konst1,
                konst_nchw,
                konst_oihw,
                konst_nchw,
                c,
                null,
                null,
                null,
                null,
            ],
        )
        y3 = relay.Call(
            conv2d_op, [y2, w3, v_one, v_one, v_one, konst1, konst_nchw, konst_oihw, konst_nchw]
        )
        ret = relay.Call(
            f3,
            [
                y2,
                w2,
                v_one,
                v_zero,
                v_one,
                konst1,
                konst_nchw,
                konst_oihw,
                konst_nchw,
                y3,
                null,
                null,
            ],
        )
        return relay.Function([x, c, w1, w2, w3], ret)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = fuse_module(mod)
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], func_expected)


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand

        @mnm.model.trace
        def forward(self, x):
            pooled = mnm.max_pool2d(x, kernel=(3, 3), stride=(1, 1), padding=1)
            concat = mnm.concatenate((pooled, x), axis=1)
            return mnm.add(concat, self.c)

    def expected(shape):
        max_pool2d_op = mnm._ffi.op.GetOp("mnm.op.max_pool2d")
        concat_op = mnm._ffi.op.GetOp("mnm.op.tvm.concatenate")
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        konst1 = mnm.ir.const(1)
        konst3 = mnm.ir.const(3)
        knchw = mnm.ir.const("NCHW")
        true = mnm.ir.const(True)
        false = mnm.ir.const(False)
        null = mnm.ir.const(None)

        p0 = mnm.ir.var("p", shape=shape)
        p1 = mnm.ir.var("p", shape=shape)
        p2 = mnm.ir.var("p", shape=(1,))
        p3 = mnm.ir.var("p", relay.TupleType(()))
        p4 = mnm.ir.var("p", relay.TupleType(()))
        concat = relay.Call(concat_op, [relay.Tuple([p0, p1]), konst1])
        out = relay.Call(add_op, [concat, p2, p3, p4])
        f2 = relay.Function([p0, p1, p2, p3, p4], out)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f2 = f2.with_attr("Dialect", "tvm")

        x = mnm.ir.var("x", shape=shape)
        c = mnm.ir.var("c", shape=(1,))
        y1 = relay.Call(max_pool2d_op, [x, konst3, konst1, konst1, konst1, false, true, knchw])
        y2 = relay.Call(f2, [y1, x, c, null, null])
        return relay.Function([x, c], y2)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    after = fuse_module(before)
    func_expected = run_infer_type(expected((1, 16, 64, 64)))
    assert tvm.ir.structural_equal(after["main"], func_expected)


def test_tuple_root_fuse():
    """Test fusion case where Tuple node is the root in its group"""
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand

        @mnm.model.trace
        def forward(self, x):
            pooled = mnm.max_pool2d(x, kernel=(3, 3), stride=1, padding=1)
            return (mnm.add(pooled, self.c), x)

    def expected(shape):
        max_pool2d_op = mnm._ffi.op.GetOp("mnm.op.tvm.max_pool2d")
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        v_three = mnm.ir.const([3, 3], dtype="int32")
        v_one = mnm.ir.const([1])
        knchw = mnm.ir.const("NCHW")
        true = mnm.ir.const(True)
        false = mnm.ir.const(False)
        null = mnm.ir.const(None)

        p0 = mnm.ir.var("p0", shape=shape)
        p1 = mnm.ir.var(
            "p1", relay.TupleType((relay.TensorType((), "int32"), relay.TensorType((), "int32")))
        )
        p2 = mnm.ir.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = mnm.ir.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = mnm.ir.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = mnm.ir.var("p5", "bool")
        p6 = mnm.ir.var("p6", "bool")
        p7 = mnm.ir.var("p7", "int64")
        p8 = mnm.ir.var("p8", relay.TupleType(()))
        p9 = mnm.ir.var("p9", relay.TupleType(()))
        c = mnm.ir.var("c", shape=(1,))

        pooled = relay.Call(max_pool2d_op, [p0, p1, p2, p3, p4, p5, p6, p7])
        out = relay.Call(add_op, [pooled, c, p8, p9])
        f = relay.Function([p0, p1, p2, p3, p4, p5, p6, p7, c, p8, p9], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "tvm")

        x = mnm.ir.var("x", shape=shape)
        c = mnm.ir.var("c", shape=(1,))
        y = relay.Call(f, [x, v_three, v_one, v_one, v_one, false, true, knchw, c, null, null])
        y = relay.Tuple([y, x])
        return relay.Function([x, c], y)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    after = fuse_module(before)
    func_expected = expected((1, 16, 64, 64))
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(after["main"], func_expected)


def test_tuple_root_no_fuse():
    """Test no fusion case where Tuple node is the root in its group."""

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.concatenate((a, b, c))

    model = Model()
    m_a, _ = randn((128,))
    m_b, _ = randn((128,))
    m_c, _ = randn((128,))
    before = model._internal(m_a, m_b, m_c).mod
    after = fuse_module(before)
    # need to convert before into graph normal form
    before = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(before))
    # The group of tuple and concatenate won't be fused due to no call node,
    # so fusion pass has no effect in this case.
    assert tvm.ir.structural_equal(after["main"], before["main"])


def test_single_w_tuple():
    """Call nodes that cannot be fused should not be in a function."""
    shape = [32, 3, 224, 224]
    momentum = 0.1
    eps = 1e-5
    stats_shape = [shape[1]]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_x, m_w, m_b, m_m, m_v):
            res = mnm.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            y = res[0]
            y = mnm.relu(y)
            new_m = res[1]
            new_v = res[2]
            return (y, new_m, new_v)

    model = Model()
    m_x, _ = randn(shape)
    m_m, _ = randn(stats_shape)
    m_v, _ = randn(stats_shape, positive=True)
    m_w, _ = randn(stats_shape)
    m_b, _ = randn(stats_shape)
    before = model._internal(m_x, m_w, m_b, m_m, m_v).mod
    after = fuse_module(before)
    # need to convert before into graph normal form
    before = run_infer_type(mnm._ffi.pass_.ToGraphNormalForm()(before))
    # BatchNorm and ReLU cannot be fused together and each of them
    # will not form a function, so fusion should have no effect in this case.
    assert tvm.ir.structural_equal(after["main"], before["main"])


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_fuse_with_dialect():
    """Fuse TVM after fusing dialect."""
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(
                16, 16, kernel_size=(3, 3), padding=1, bias=False, channel_mode="NHWC"
            )

        @mnm.model.trace
        def forward(self, x):
            y = self.conv1(x)
            y = mnm.add(y, self.c)
            y = mnm.add(y, self.c)
            y = mnm.relu(y)
            return y

    def expected():
        v_one = mnm.ir.const([1])
        konst1 = mnm.ir.const(1)
        null = mnm.ir.const(None)
        konst_nhwc = mnm.ir.const("NHWC")
        konst_ohwi = mnm.ir.const("OHWI")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.cutlass.conv2d")
        cutlass_add_op = mnm._ffi.op.GetOp("mnm.op.cutlass.add")
        tvm_add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.tvm.relu")

        # segment
        x = mnm.ir.var("p", shape=(1, 64, 64, 16))
        w = mnm.ir.var("p", shape=(16, 3, 3, 16))
        p2 = mnm.ir.var("p", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = mnm.ir.var("p", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = mnm.ir.var("p", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = mnm.ir.var("p", "int64")
        p6 = mnm.ir.var("p", "int64")
        p7 = mnm.ir.var("p", "int64")
        p8 = mnm.ir.var("p", "int64")
        p9 = mnm.ir.var("p", shape=(1,))
        p10 = mnm.ir.var("p", relay.TupleType(()))
        p11 = mnm.ir.var("p", relay.TupleType(()))
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y = relay.Call(cutlass_add_op, [y, p9, p10, p11])
        f1 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f1 = f1.with_attr("Dialect", "cutlass")
        f1 = f1.with_attr("PatternName", "conv2d_fusion")

        p0 = mnm.ir.var("p", shape=(1, 64, 64, 16))
        p1 = mnm.ir.var("p", shape=(1,))
        p2 = mnm.ir.var("p", relay.TupleType(()))
        p3 = mnm.ir.var("p", relay.TupleType(()))
        y = relay.Call(tvm_add_op, [p0, p1, p2, p3])
        y = relay.Call(relu_op, [y])
        f2 = relay.Function([p0, p1, p2, p3], y)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f2 = f2.with_attr("Dialect", "tvm")

        x = mnm.ir.var("p0", shape=(1, 64, 64, 16))
        w = mnm.ir.var("p1", shape=(16, 3, 3, 16))
        c = mnm.ir.var("c", shape=(1,))
        y = relay.Call(
            f1,
            [x, w, v_one, v_one, v_one, konst1, konst_nhwc, konst_ohwi, konst_nhwc, c, null, null],
        )
        out = relay.Call(f2, [y, c, null, null])

        return relay.Function([x, c, w], out)

    model = Model()
    m_x, _ = randn((1, 64, 64, 16), device="cpu")
    mod = model._internal(m_x).mod
    with mnm.device("cuda"):
        mod = fuse_module(mod, True)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod["main"], func_expected)


# TODO@(hzfan): fix issue #402
@pytest.mark.xfail
def test_sgd():
    shape = [2, 3, 4]
    dtype = "float32"
    device = "llvm"

    class Model(mnm.Model):
        def build(self):
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*shape).astype(dtype), device=device)

        @mnm.model.trace
        def forward(self, dy):
            y = mnm.relu(self.x)
            dx = mnm.relu_dx(self.x, y, dy)
            return y, dx

    class SGD(mnm.Model):
        def build(self, model, lr=0.1, mu=0.01):
            self.model = model
            self.lr = lr
            self.mu = mu
            self.reset()

        def reset(self):
            self.v = mnm.array(np.zeros(shape, dtype=dtype), device=device)
            self.model.reset()

        @mnm.model.trace
        def forward(self, dy):
            out = self.model(dy)
            y = out[0]
            dx = out[1]
            # update params
            sgd_out = mnm.sgd(self.model.x, dx, self.v, self.lr, self.mu)
            new_v = sgd_out[0]
            new_x = sgd_out[1]
            trace_mutate_attr(self.model, "x", new_x)
            trace_mutate_attr(self, "v", new_v)
            return y

    def expected():
        relu_op = mnm._ffi.op.GetOp("mnm.op.tvm.relu")
        relu_dx_op = mnm._ffi.op.GetOp("mnm.op.tvm.relu_dx")
        sgd_op = mnm._ffi.op.GetOp("mnm.op.tvm.sgd")
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = mnm.ir.var("p0", shape=shape)
        dy = mnm.ir.var("p1", shape=shape)
        v = mnm.ir.var("p2", shape=shape)
        y = relay.Call(relu_op, [x])
        y1 = relay.Call(relu_dx_op, [x, y, dy])
        y2 = relay.Call(sgd_op, [x, y1, v, default, default])
        out = relay.Tuple([y, relay.TupleGetItem(y2, 1), relay.TupleGetItem(y2, 0)])
        f = relay.Function([x, dy, v], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "tvm")

        x = mnm.ir.var("model.x", shape=shape)
        dy = mnm.ir.var("dy", shape=shape)
        v = mnm.ir.var("v", shape=shape)
        y = relay.Call(f, [x, dy, v])
        return relay.Function([dy, v, x], y)

    m_param, _ = randn(shape, device=device)
    n_v = np.zeros(shape, dtype=dtype)
    m_dy, _ = randn(shape, device=device)
    model = Model()
    sgd = SGD(model)
    model.x = m_param
    sgd.v = mnm.array(n_v, device=device)
    mod = sgd._internal(m_dy).mod
    mod = fuse_module(mod)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod["main"], func_expected)


def test_fuse_inplace():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self, shape):
            self.c = konst
            self.shape = shape

        @mnm.model.trace
        def forward(self, x, y):
            y = mnm.add(y, self.c)
            new_x = mnm.add(x, y, out=x)
            z = mnm.relu(new_x)
            return z

    def expected(shape):
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        null = mnm.ir.const(None)

        p0 = mnm.ir.var("p0", shape=shape)
        p1 = mnm.ir.var("p1", shape=shape)
        p2 = mnm.ir.var("p2", shape=(1,))
        p3 = mnm.ir.var("p", relay.TupleType(()))
        p4 = mnm.ir.var("p", relay.TupleType(()))
        p5 = mnm.ir.var("p", relay.TupleType(()))
        y = relay.Call(add_op, [p1, p2, p3, p4])
        y = relay.Call(add_op, [p0, y, p0, p5])
        f = relay.Function([p0, p1, p2, p3, p4, p5], y)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        f = f.with_attr("Dialect", "tvm")

        x = mnm.ir.var("x", shape=shape)
        y = mnm.ir.var("y", shape=shape)
        c = mnm.ir.var("c", shape=(1,))
        out = relay.Call(f, [x, y, c, null, null, null])
        out = relay.Call(relu_op, [out])
        return relay.Function([x, y, c], out)

    shape = (10, 20)
    model = Model(shape)
    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    mod_before = model._internal(m_x, m_y).mod
    mod_after = fuse_module(mod_before)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod_after["main"], func_expected)


def test_may_share():
    """Use SGD optimizer updating logic to test the IR with may_share.
    TODO(issue 758): Rewrite this testcase to directly use may_share API in frontend.
    """
    shape = (5, 2)

    def before():
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        sub_op = mnm._ffi.op.GetOp("mnm.op.subtract")
        mul_op = mnm._ffi.op.GetOp("mnm.op.multiply")

        data_w = mnm.ir.var("w", shape=shape, dtype="float32")
        data_v = mnm.ir.var("v", shape=shape, dtype="float32")
        data_g = mnm.ir.var("g", shape=shape, dtype="float32")

        const_m = mnm.ir.const(0.2, dtype="float32")
        const_lr = mnm.ir.const(0.05, dtype="float32")
        null = mnm.ir.const(None)

        # Note that we do not use the "out" arguments in add and sub to represent may_share.
        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(mul_op, [const_m, data_v]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, data_g, null, null]), may_share=data_v)
        a_3 = sb.let("a3", relay.Call(mul_op, [const_lr, a_2]))
        a_4 = sb.let("a4", relay.Call(sub_op, [data_w, a_3, null, null]), may_share=data_w)

        sb.ret(a_4)
        func = relay.Function([data_w, data_v, data_g], sb.get())
        mod = tvm.IRModule.from_expr(func)
        return mod

    def expected():
        add_op = mnm._ffi.op.GetOp("mnm.op.tvm.add")
        sub_op = mnm._ffi.op.GetOp("mnm.op.tvm.subtract")
        mul_op = mnm._ffi.op.GetOp("mnm.op.tvm.multiply")

        # Fused 1
        const_m = mnm.ir.const(0.2, dtype="float32")
        p_0 = mnm.ir.var("p0", shape=shape, dtype="float32")
        p_1 = mnm.ir.var("p1", shape=shape, dtype="float32")
        p_2 = mnm.ir.var("p2", relay.TupleType(()))
        out = relay.Call(mul_op, [const_m, p_0])
        out = relay.Call(add_op, [out, p_1, p_2, p_2])
        func1 = relay.Function([p_0, p_1, p_2], out)
        func1 = func1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func1 = func1.with_attr("Dialect", "tvm")

        # Fuse 2
        const_lr = mnm.ir.const(0.05, dtype="float32")
        p_01 = mnm.ir.var("p01", shape=shape, dtype="float32")
        p_11 = mnm.ir.var("p11", shape=shape, dtype="float32")
        p_21 = mnm.ir.var("p2", relay.TupleType(()))
        out = relay.Call(mul_op, [const_lr, p_11])
        out = relay.Call(sub_op, [p_01, out, p_21, p_21])
        func2 = relay.Function([p_01, p_11, p_21], out)
        func2 = func2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func2 = func2.with_attr("Dialect", "tvm")

        # Main function
        data_w = mnm.ir.var("w", shape=shape, dtype="float32")
        data_v = mnm.ir.var("v", shape=shape, dtype="float32")
        data_g = mnm.ir.var("g", shape=shape, dtype="float32")

        const_m = mnm.ir.const(0.2, dtype="float32")
        const_lr = mnm.ir.const(0.05, dtype="float32")
        null = mnm.ir.const(None)

        sb = ScopeBuilder()
        a_2 = sb.let("a2", relay.Call(func1, [data_v, data_g, null]), may_share=data_v)
        a_4 = sb.let("a4", relay.Call(func2, [data_w, a_2, null]), may_share=data_w)
        sb.ret(a_4)

        return relay.Function([data_w, data_v, data_g], sb.get())

    mod_after = fuse_module(before())
    func_expected = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod_after["main"], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
