# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,no-self-use
import numpy as np
import pytest
import mnm
from mnm.model import Conv2d
from mnm.model.trace import trace_mutate_attr
from mnm.testing import run_infer_type, randn
import tvm
from tvm import relay


def fuse_module(mod, fuse_level=3):
    with mnm.ir.PassContext(config={"mnm.fuse_level": fuse_level}):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseOps()(mod)
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
        x = mnm.ir.var("p0", shape=shape)
        y = mnm.ir.var("p1", shape=(1,))
        z = mnm.ir.op.add(x, y)
        z = mnm.ir.op.log(mnm.ir.op.relu(z))
        f1 = relay.Function([x, y], z)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=shape)
        y = relay.var("c", shape=(1,))
        ret = relay.Call(f1, [x, y])
        return relay.Function([x, y], ret)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    mod = fuse_module(mod)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod['main'], func_expected)


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

        # segment 1
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 3, 3))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "int64")
        p6 = relay.var("p6", "int64")
        p7 = relay.var("p7", "int64")
        p8 = relay.var("p8", "int64")
        p9 = relay.var("p9", shape=(1,))
        y = mnm.ir.op.conv2d(x, w, p2, p3, p4, p5, p6, p7, p8)
        y1 = mnm.ir.op.add(y, p9)
        y = mnm.ir.op.add(y, y1)
        f1 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p9], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 3
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 1, 1))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "int64")
        p6 = relay.var("p6", "int64")
        p7 = relay.var("p7", "int64")
        p8 = relay.var("p8", "int64")
        p91 = relay.var("p91", shape=(1, 16, 64, 64))
        y = mnm.ir.op.conv2d(x, w, p2, p3, p4, p5, p6, p7, p8)
        y = mnm.ir.op.add(y, p91)
        f3 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p91], y)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # compose
        x = relay.var("x", shape=(1, 16, 64, 64))
        c = relay.var("c", shape=(1,))
        w1 = relay.var("conv1.w", shape=(16, 16, 3, 3))
        w2 = relay.var("conv2.w", shape=(16, 16, 1, 1))
        w3 = relay.var("conv3.w", shape=(16, 16, 3, 3))
        y1 = mnm.ir.op.add(x, c)
        y2 = relay.Call(f1, [y1, w1, v_one, v_one, v_one, konst1, konst_nchw,
                             konst_oihw, konst_nchw, c])
        y3 = mnm.ir.op.conv2d(y2, w3, 1, 1)
        ret = relay.Call(f3, [y2, w2, v_one, v_zero, v_one, konst1,
                              konst_nchw, konst_oihw, konst_nchw, y3])
        return relay.Function([x, c, w1, w2, w3], ret)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = fuse_module(mod)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod['main'], func_expected)


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

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p0", shape=shape)
        p2 = relay.var("p2", shape=(1,))
        concat = mnm.ir.op.concatenate([p0, p1], 1)
        out = mnm.ir.op.add(concat, p2)
        f2 = relay.Function([p0, p1, p2], out)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        y1 = mnm.ir.op.max_pool2d(x, 3, 1, 1)
        y2 = relay.Call(f2, [y1, x, c])
        return relay.Function([x, c], y2)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    after = fuse_module(before)
    func_expected = run_infer_type(expected((1, 16, 64, 64)))
    assert tvm.ir.structural_equal(after['main'], func_expected)


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
        v_three = mnm.ir.const([3, 3], dtype="int32")
        v_one = mnm.ir.const([1])
        knchw = mnm.ir.const("NCHW")
        true = mnm.ir.const(True)
        false = mnm.ir.const(False)

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p1", relay.TupleType(
            (relay.TensorType((), "int32"), relay.TensorType((), "int32"))))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "bool")
        p6 = relay.var("p6", "bool")
        p7 = relay.var("p7", "int64")
        c = relay.var("c", shape=(1,))

        pooled = mnm.ir.op.max_pool2d(p0, p1, p2, p3, p4, p5, p6, p7)
        out = mnm.ir.op.add(pooled, c)
        f = relay.Function([p0, p1, p2, p3, p4, p5, p6, p7, c], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        y = relay.Call(f, [x, v_three, v_one, v_one, v_one, false, true, knchw, c])
        y = relay.Tuple([y, x])
        return relay.Function([x, c], y)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    after = fuse_module(before)
    func_expected = expected((1, 16, 64, 64))
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(after['main'], func_expected)


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


def test_fuse_level_1():
    """Fuse level 1 only fuses injective nodes."""
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @mnm.model.trace
        def forward(self, x):
            y = self.conv1(x)
            y1 = mnm.add(y, self.c)
            y = mnm.add(y, y1)
            return y

    def expected():
        # segment
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 3, 3))
        conv_out = mnm.ir.op.conv2d(x, w, 1, 1)

        p0 = relay.var("p0", shape=(1, 16, 64, 64))
        p1 = relay.var("p1", shape=(1,))
        y = mnm.ir.op.add(p0, p1)
        y = mnm.ir.op.add(p0, y)
        f1 = relay.Function([p0, p1], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        c = relay.var("c", shape=(1,))
        out = relay.Call(f1, [conv_out, c])

        return relay.Function([x, c, w], out)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    mod = fuse_module(mod, fuse_level=1)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod["main"], func_expected)


# TODO@(hzfan): fix issue #402
@pytest.mark.xfail
def test_sgd():
    shape = [2, 3, 4]
    dtype = 'float32'
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
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = relay.var("p0", shape=shape)
        dy = relay.var("p1", shape=shape)
        v = relay.var("p2", shape=shape)
        y = mnm.ir.op.relu(x)
        y1 = mnm.ir.op.relu_dx(x, y, dy)
        y2 = mnm.ir.op.sgd(x, y1, v, default, default)
        out = relay.Tuple([y, relay.TupleGetItem(y2, 1), relay.TupleGetItem(y2, 0)])
        f = relay.Function([x, dy, v], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("model.x", shape=shape)
        dy = relay.var("dy", shape=shape)
        v = relay.var("v", shape=shape)
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
    assert tvm.ir.structural_equal(mod['main'], func_expected)


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
        p0 = mnm.ir.var("p0", shape=shape)
        p1 = mnm.ir.var("p1", shape=shape)
        p2 = mnm.ir.var("p2", shape=(1,))
        y = mnm.ir.op.add(p1, p2)
        y = mnm.ir.op.add(p0, y, p0)
        f = relay.Function([p0, p1, p2], y)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = mnm.ir.var("x", shape=shape)
        y = mnm.ir.var("y", shape=shape)
        c = mnm.ir.var("c", shape=(1,))
        out = relay.Call(f, [x, y, c])
        out = mnm.ir.op.relu(out)
        return relay.Function([x, y, c], out)

    shape = (10, 20)
    model = Model(shape)
    m_x, _ = randn(shape, device="cpu")
    m_y, _ = randn(shape, device="cpu")
    mod_before = model._internal(m_x, m_y).mod
    mod_after = fuse_module(mod_before)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod_after['main'], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
