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
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        log_op = mnm._ffi.op.GetOp("mnm.op.log")
        default = mnm.ir.const(None)

        x = relay.var("p0", shape=shape)
        y = relay.var("p1", shape=(1,))
        z = relay.Call(add_op, [x, y, default, default])
        z = relay.Call(log_op, [relay.Call(relu_op, [z])])
        f1 = relay.Function([x, y], z)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=shape)
        y = relay.var("c", shape=(1,))
        z = relay.var("a3")
        ret = relay.Let(z, relay.Call(f1, [x, y]), z)
        return relay.Function([x, y], ret)

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod_before = model._internal(m_x).mod
    mod_before = run_infer_type(mod_before)
    mod_after = mnm._ffi.pass_.FuseOps(3)(mod_before)
    mod_after = run_infer_type(mod_after)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(mod_after['main'], func_expected)


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
        default = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")

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
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y1 = relay.Call(add_op, [y, p9, default, default])
        y = relay.Call(add_op, [y, y1, default, default])
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
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y = relay.Call(add_op, [y, p91, default, default])
        f3 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, p91], y)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # compose
        x = relay.var("x", shape=(1, 16, 64, 64))
        c = relay.var("c", shape=(1,))
        w1 = relay.var("conv1.w", shape=(16, 16, 3, 3))
        w2 = relay.var("conv2.w", shape=(16, 16, 1, 1))
        w3 = relay.var("conv3.w", shape=(16, 16, 3, 3))
        a1 = relay.var("a1")
        a4 = relay.var("a4")
        a6 = relay.var("a6")
        a7 = relay.var("a7")
        let3 = relay.Let(a7, relay.Call(f3, [a4, w2, v_one, v_zero, v_one, konst1,
                                             konst_nchw, konst_oihw, konst_nchw, a6]), a7)
        let2 = relay.Let(a6, relay.Call(conv2d_op, [a4, w3, v_one, v_one, v_one, konst1,
                                                    konst_nchw, konst_oihw, konst_nchw]), let3)
        let1 = relay.Let(a4, relay.Call(f1, [a1, w1, v_one, v_one, v_one, konst1,
                                             konst_nchw, konst_oihw, konst_nchw, c]), let2)
        let = relay.Let(a1, relay.Call(add_op, [x, c, default, default]), let1)
        return relay.Function([x, c, w1, w2, w3], let)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod_before = model._internal(m_x).mod
    mod_before = run_infer_type(mod_before)
    mod_after = mnm._ffi.pass_.FuseOps(3)(mod_before)
    mod_after = run_infer_type(mod_after)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod_after['main'], func_expected)


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
        concat_op = mnm._ffi.op.GetOp("mnm.op.concatenate")
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        konst1 = mnm.ir.const(1)
        konst3 = mnm.ir.const(3)
        knchw = mnm.ir.const("NCHW")
        true = mnm.ir.const(True)
        false = mnm.ir.const(False)
        default = mnm.ir.const(None)

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p0", shape=shape)
        p2 = relay.var("p2", shape=(1,))
        concat = relay.Call(concat_op, [relay.Tuple([p0, p1]), konst1])
        out = relay.Call(add_op, [concat, p2, default, default])
        f2 = relay.Function([p0, p1, p2], out)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        a1 = relay.var("a1")
        a4 = relay.var("a4")
        let2 = relay.Let(a4, relay.Call(f2, [a1, x, c]), a4)
        let1 = relay.Let(a1, relay.Call(max_pool2d_op, [x, konst3, konst1, konst1, konst1,
                                                        false, true, knchw]), let2)
        return relay.Function([x, c], let1)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(3)(before)
    after = run_infer_type(after)
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
        max_pool2d_op = mnm._ffi.op.GetOp("mnm.op.max_pool2d")
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        v_three = mnm.ir.const([3, 3], dtype="int32")
        v_one = mnm.ir.const([1])
        knchw = mnm.ir.const("NCHW")
        true = mnm.ir.const(True)
        false = mnm.ir.const(False)
        default = mnm.ir.const(None)

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

        pooled = relay.Call(max_pool2d_op, [p0, p1, p2, p3, p4, p5, p6, p7])
        out = relay.Call(add_op, [pooled, c, default, default])
        f = relay.Function([p0, p1, p2, p3, p4, p5, p6, p7, c], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        let3 = relay.Let(a3, relay.Tuple([a2, x]), a3)
        let2 = relay.Let(
            a2, relay.Call(f, [x, v_three, v_one, v_one, v_one, false, true, knchw, c]), let3)
        return relay.Function([x, c], let2)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    before = model._internal(m_x).mod
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(3)(before)
    after = run_infer_type(after)
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
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(3)(before)
    after = run_infer_type(after)
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
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(3)(before)
    after = run_infer_type(after)
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
        v_one = mnm.ir.const([1])
        konst1 = mnm.ir.const(1)
        konst_nchw = mnm.ir.const("NCHW")
        konst_oihw = mnm.ir.const("OIHW")
        default = mnm.ir.const(None)
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        a4 = relay.var("a4")
        a6 = relay.var("a6")

        # segment
        x = relay.var("p0", shape=(1, 16, 64, 64))
        c = relay.var("c", shape=(1,))
        y = relay.Call(add_op, [x, c, default, default])
        y = relay.Call(add_op, [x, y, default, default])
        f1 = relay.Function([x, c], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        c = relay.var("c", shape=(1,))
        let2 = relay.Let(a4, relay.Call(f1, [a6, c]), a4)

        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 3, 3))
        let1 = relay.Let(a6, relay.Call(conv2d_op, [x, w, v_one, v_one, v_one, konst1,
                                                    konst_nchw, konst_oihw, konst_nchw]), let2)
        return relay.Function([x, c, w], let1)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod_before = model._internal(m_x).mod
    mod_before = run_infer_type(mod_before)
    mod_after = mnm._ffi.pass_.FuseOps(1)(mod_before)
    mod_after = run_infer_type(mod_after)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod_after["main"], func_expected)

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
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        relu_dx_op = mnm._ffi.op.GetOp("mnm.op.relu_dx")
        sgd_op = mnm._ffi.op.GetOp("mnm.op.sgd")
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = relay.var("p0", shape=shape)
        dy = relay.var("p1", shape=shape)
        v = relay.var("p2", shape=shape)
        y = relay.Call(relu_op, [x])
        y1 = relay.Call(relu_dx_op, [x, y, dy])
        y2 = relay.Call(sgd_op, [x, y1, v, default, default])
        out = relay.Tuple([y, relay.TupleGetItem(y2, 1), relay.TupleGetItem(y2, 0)])
        f = relay.Function([x, dy, v], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("model.x", shape=shape)
        dy = relay.var("dy", shape=shape)
        v = relay.var("v", shape=shape)
        a6 = relay.var("a6")
        let = relay.Let(a6, relay.Call(f, [x, dy, v]), a6)
        return relay.Function([dy, v, x], let)

    m_param, _ = randn(shape, device=device)
    n_v = np.zeros(shape, dtype=dtype)
    m_dy, _ = randn(shape, device=device)
    model = Model()
    sgd = SGD(model)
    model.x = m_param
    sgd.v = mnm.array(n_v, device=device)
    mod = sgd._internal(m_dy).mod
    mod = run_infer_type(mod)
    mod = mnm._ffi.pass_.FuseOps(3)(mod)
    mod = run_infer_type(mod)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(mod['main'], func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
