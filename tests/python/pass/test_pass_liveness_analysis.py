# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, no-self-use, too-many-locals, unused-variable, protected-access
# pylint: disable=too-many-arguments
import pytest
import raf
from raf._lib import tvm, relay
from raf.ir import ScopeBuilder
from raf._ffi.pass_ import InferType, LivenessAnalysis, ManifestAlloc, FoldConstant
from raf.testing import randn


def verify_live_in_set(mod, expected):
    mod = InferType()(mod)

    # Check liveness analysis result.
    ret = LivenessAnalysis(mod)
    ret = {key.name_hint: {v.name_hint for v in var_list} for key, var_list in ret.items()}

    missed = {}
    for key, vset in expected.items():
        if key not in ret:
            missed[key] = []
        else:
            for var in ret[key]:
                if var not in vset:
                    if key not in missed:
                        missed[key] = []
                    missed[key].append(var)

    if missed or not expected:
        print("IR:\n%s" % raf.ir.AsText(mod))
        print("Live in sets:")
        for key, var_list in ret.items():
            print("%s: %s" % (key, ",".join(var_list)))

        print("\nMissed items")
        for key, var_list in missed.items():
            if not var_list:
                print("Missed key %s" % key)
            else:
                print("Missed live in of %s: %s" % (key, ",".join(var_list)))
        assert False, "Expected key or live in vars are missing"


def test_basic():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2, param_3):
            t_0 = raf.add(param_0, param_0)  # a1
            t_1 = raf.add(t_0, param_1)  # a2
            t_2 = raf.add(t_1, param_2)  # a3
            t_3 = raf.add(t_2, t_0)  # a4
            t_4 = raf.add(t_3, param_3)  # a5
            return t_4  # n_1

    device = "cpu"
    shape = (5, 5)
    model = Model()
    model.infer_mode()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    m_d, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c, m_d]

    expected = {
        "n_0": {},
        "a1": {"param_0", "param_1", "param_2", "param_3"},
        "a2": {"t_0", "param_1", "param_2", "param_3"},
        "a3": {"t_0", "t_1", "param_2", "param_3"},
        "a4": {"t_0", "t_2", "param_3"},
        "a5": {"t_3", "param_3"},
        "n_1": {"t_4"},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_multi_outs():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2, param_3, param_4):
            t_0 = raf.relu(param_0)  # a1
            res = raf.batch_norm_train(t_0, param_3, param_4, param_1, param_2, 0.1, 1e-5)  # a2
            t_1 = res[0]  # a3
            t_2 = res[1]
            t_3 = res[2]
            t_4 = raf.relu(t_1)  # a4
            t_5 = raf.relu(t_4)  # a5
            return t_5  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (16, 3, 224, 224)
    stats_shape = [shape[1]]
    m_x, _ = randn(shape, device=device)
    m_m, _ = randn(stats_shape, device=device)
    m_v, _ = randn(stats_shape, positive=True, device=device)
    m_w, _ = randn(stats_shape, device=device)
    m_b, _ = randn(stats_shape, device=device)
    args = [m_x, m_m, m_v, m_w, m_b]

    expected = {
        "n_0": {},
        "a1": {"param_0", "param_1", "param_2", "param_3", "param_4"},
        "a2": {"t_0", "param_1", "param_2", "param_3", "param_4"},
        "a3": {"t_1"},
        "a4": {"t_1"},
        "a5": {"t_4"},
        "n_1": {"t_5"},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_tuple_input():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, tup):
            x = tup[0]  # a1
            y = tup[1]  # a2
            t_0 = raf.add(x, y)  # a3
            return t_0  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    args = [(m_a, m_b)]

    expected = {
        "n_0": {},
        "a1": {"param_0", "param_1"},
        "a2": {"param_0", "param_1"},
        "a3": {"param_0", "param_1"},
        "n_1": {"t_0"},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_unused_tuple():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, tup):
            x = tup[0]  # a1
            t_0 = raf.add(x, x)  # a2
            t_1 = raf.concatenate(tup)  # a3
            ret = (t_0, t_1)  # a4
            return ret  # n_1

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    args = [(m_a, m_b)]

    # There won't be a tgi_1 because it is never be used.
    expected = {
        "n_0": {},
        "a1": {"param_0", "param_1"},
        "a2": {"param_0", "param_1"},
        "a3": {"param_0", "param_1", "t_0"},
        "a4": {"t_0", "t_1"},
        "n_1": {"t_0", "t_1"},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_direct_assign():
    sb = ScopeBuilder()
    p0 = raf.ir.var("p0", shape=(10, 10))
    a_1 = sb.let("a1", raf.ir.op.relu(p0))
    a_2 = sb.let("a2", a_1)
    a_3 = sb.let("a3", raf.ir.op.relu(a_2))
    sb.ret(a_3)
    mod = tvm.IRModule.from_expr(relay.Function([p0], sb.get()))

    expected = {
        "n_0": {},
        "a1": {"param_0"},
        "a2": {"t_0"},
        "a3": {"t_0"},
        "n_1": {"t_1"},
    }
    verify_live_in_set(mod, expected)


def test_folded_const_in_tuple():
    shape = (5, 5)

    sb = ScopeBuilder()
    p0 = raf.ir.var("p0", shape=shape)
    a1 = sb.let("a1", raf.ir.op.relu(p0))
    x0 = sb.let("x0", raf.ir.op.zeros(raf.ir.const(shape), raf.ir.const("float32")))
    # x0 will be folded and we should have "let %a2 = (%p0, %a1, tensor(5x5, float32, cpu(0)))"
    a2 = sb.let("a2", relay.Tuple([p0, a1, x0]))
    a3 = sb.let("a3", relay.TupleGetItem(a2, 2))
    sb.ret(a3)
    mod = tvm.IRModule.from_expr(relay.Function([p0], sb.get()))
    mod = InferType()(mod)
    mod = FoldConstant()(mod)
    mod = InferType()(mod)

    expected = {
        "n_0": {},
        "a1": {"param_0"},
        "a2": {"param_0", "t_0"},
        "a3": {"t_1"},  # t_1 is the folded const generated at %a2
        "n_1": {"t_1"},
    }
    verify_live_in_set(mod, expected)


def test_reshape():
    shape = (10, 10)

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            t_0 = raf.relu(x)
            t_1 = raf.reshape(t_0, (shape[0] * shape[1],))
            t_2 = raf.relu(t_1)
            return t_2

    model = Model()
    model.infer_mode()

    device = "cpu"
    m_x, _ = randn(shape, device=device)
    args = [m_x]

    expected = {
        "n_0": {},
        "a1": {"param_0"},
        "a2": {"t_0"},
        "a3": {"t_0"},
        "n_1": {"t_1"},
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_manifest_alloc_compatible():
    def test_func():
        add_op = raf._ffi.op.GetOp("raf.op.add")
        null = raf.ir.const(None)

        x = relay.var("x", shape=(5, 5))
        y = relay.var("y", shape=(5, 5))
        a0 = relay.var("a0")
        a1 = relay.var("a1")
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        a4 = relay.var("a4")
        a5 = relay.var("a5")
        a6 = relay.var("a6")
        a7 = relay.var("a7")

        let7 = relay.Let(a7, a1, a7)
        let6 = relay.Let(a6, raf.ir.op.vm_invoke_op(a2, a4, a5), let7)
        let5 = relay.Let(a5, relay.Tuple((a1,)), let6)
        # Test both binded and non-binded constants
        let4 = relay.Let(a4, relay.Tuple((x, y, a3, null)), let5)
        let3 = relay.Let(a3, null, let4)
        let2 = relay.Let(a2, add_op, let3)
        let1 = relay.Let(a1, raf.ir.op.vm_alloc_tensor(a0, [5, 5], "float32", [5, 5]), let2)
        let0 = relay.Let(a0, raf.ir.op.vm_alloc_storage(100, 64, 1, 0), let1)
        # pylint: disable=line-too-long
        # fn (%x: Tensor[(5, 5), float32], %y: Tensor[(5, 5), float32]) {
        #   let %a0 = raf.op.vm.alloc_storage(int64(100), int64(64), int64(1), int64(0), str"float32");
        #   let %a1 = raf.op.vm.alloc_tensor(%a0, TupleValue([int64(5), int64(5)]), str"float32", TupleValue([int64(5), int64(5)]));
        #   let %a2 = raf.op.add;
        #   let %a3 = nullptr;
        #   let %a4 = (%x, %y, %a3, nullptr);
        #   let %a5 = (%a1,);
        #   let %a6 = raf.op.vm.invoke_op(%a2, %a4, %a5);
        #   let %a7 = %a1;
        #   %a7
        # }
        # pylint: enable=line-too-long

        return relay.Function([x, y], let0)

    # Note that a3 will be inlined after InferType.
    expected = {
        "n_1": {},
        "a0": {"param_0", "param_1"},
        "a1": {"param_0", "param_1", "t_0"},
        "a2": {"param_0", "param_1", "t_1"},
        "a4": {"param_0", "param_1", "t_1"},
        "a5": {"param_0", "param_1", "t_1"},
        "a6": {"param_0", "param_1", "t_1"},
        "a7": {"t_1"},
        "n_2": {"t_1"},
    }

    func = test_func()
    mod = tvm.IRModule.from_expr(func)
    verify_live_in_set(mod, expected)


def test_after_manifest_alloc():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, param_0, param_1, param_2):
            t_0 = raf.add(param_0, param_0)  # a1
            t_1 = raf.add(t_0, param_1)  # a2
            t_2 = raf.add(t_1, param_2)  # a3
            return t_2  # n_1

    device = "cpu"
    shape = (5, 5)
    model = Model()
    model.infer_mode()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c]

    mod = model._internal(*args).mod
    mod = InferType()(mod)
    mod = ManifestAlloc()(mod)
    # pylint: disable=line-too-long
    # def @main(%param_0: Tensor[(5, 5), float32],
    #           %param_1: Tensor[(5, 5), float32],
    #           %param_2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x_2 = raf.op.add;
    #   let %x_3 = (%param_0, %param_0, nullptr, nullptr);
    #   let %x_4 = raf.op.vm.alloc_storage(int64(100), int64(128), int32(1), int32(0), str"float32") /* ty=float32 */;
    #   let %x_5 = raf.op.vm.alloc_tensor(%x_4, [5, 5], str"float32", [5, 5]) /* ty=Tensor[(5, 5), float32] */;
    #   let %x_6 = (%x_5,);
    #   let %x_7 = raf.op.vm.invoke_op(%x_2, %x_3, %x_6) /* ty=() */;
    #   let %a1 = %x_5;
    #   let %x_10 = raf.op.add;
    #   let %x_11 = (%a1, %param_1, nullptr, nullptr);
    #   let %x_12 = raf.op.vm.alloc_storage(int64(100), int64(128), int32(1), int32(0), str"float32") /* ty=float32 */;
    #   let %x_13 = raf.op.vm.alloc_tensor(%x_12, [5, 5], str"float32", [5, 5]) /* ty=Tensor[(5, 5), float32] */;
    #   let %x_14 = (%x_13,);
    #   let %x_15 = raf.op.vm.invoke_op(%x_10, %x_11, %x_14) /* ty=() */;
    #   let %a2 = %x_13;
    #   let %x_18 = raf.op.add;
    #   let %x_19 = (%a2, %param_2, nullptr, nullptr);
    #   let %x_20 = raf.op.vm.alloc_storage(int64(100), int64(128), int32(1), int32(0), str"float32") /* ty=float32 */;
    #   let %x_21 = raf.op.vm.alloc_tensor(%x_20, [5, 5], str"float32", [5, 5]) /* ty=Tensor[(5, 5), float32] */;
    #   let %x_22 = (%x_21,);
    #   let %x_23 = raf.op.vm.invoke_op(%x_18, %x_19, %x_22) /* ty=() */;
    #   let %a3 = %x_21;
    #   %a3
    # }
    # pylint: enable=line-too-long

    expected = {
        "x_2": {"param_0", "param_1", "param_2"},
        "x_3": {"param_0", "param_1", "param_2"},
        "x_4": {"param_0", "param_1", "param_2"},
        "x_5": {"param_0", "param_1", "t_0", "param_2"},
        "x_6": {"param_0", "param_1", "t_1", "param_2"},
        "x_7": {"param_0", "param_1", "t_1", "param_2"},
        "a1": {"param_1", "t_1", "param_2"},
        "x_10": {"param_1", "t_1", "param_2"},
        "x_11": {"param_1", "t_1", "param_2"},
        "x_12": {"param_1", "t_1", "param_2"},
        "x_13": {"param_1", "t_1", "param_2", "t_2"},
        "x_14": {"param_1", "t_1", "param_2", "t_3"},
        "x_15": {"param_1", "t_1", "param_2", "t_3"},
        "a2": {"t_3", "param_2"},
        "x_18": {"t_3", "param_2"},
        "x_19": {"t_3", "param_2"},
        "x_20": {"param_2", "t_3"},
        "x_21": {"param_2", "t_4", "t_3"},
        "x_22": {"param_2", "t_5", "t_3"},
        "x_23": {"param_2", "t_5", "t_3"},
        "a3": {"t_5"},
        "n_4": {"t_5"},
    }

    verify_live_in_set(mod, expected)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_fuse_closure():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, p0, p1, p2):
            t_0 = raf.matmul(p0, p1)
            t_1 = raf.multiply(t_0, p2)
            t_2 = raf.relu(t_1)
            return t_2

    model = Model()
    model.infer_mode()

    device = "cpu"
    shape = (5, 5)
    m_p0, _ = randn(shape, device=device)
    m_p1, _ = randn(shape, device=device)
    m_p2, _ = randn(shape, device=device)
    args = [m_p0, m_p1, m_p2]

    mod = model._internal(*args).mod
    with raf.device("cuda"):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        mod = raf._ffi.pass_.FuseTVM()(mod)
        mod = raf._ffi.pass_.ToANormalForm()(mod)
        mod = raf._ffi.pass_.InlinePrimitives()(mod)
    # fn (%p0: Tensor[(5, 5), float32],
    #     %p1: Tensor[(5, 5), float32],
    #     %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x1 = raf.op.cublas.matmul(%p0, %p1) /* ty=Tensor[(5, 5), float32] */;
    #   %1 = fn (%p01: Tensor[(5, 5), float32], %p11: Tensor[(5, 5), float32],
    #            Primitive=1, Dialect="tvm") -> Tensor[(5, 5), float32] {
    #     %0 = raf.op.tvm.multiply(%p01, %p11);
    #     raf.op.tvm.relu(%0)
    #   };
    #   let %x3 = %1(%x1, %p2);
    #   %x3
    # }
    expected = {
        "n_0": {},
        "x1": {"param_0", "param_1", "param_2"},
        "x3": {"param_2", "t_0"},
        "n_1": {"t_1"},
    }
    verify_live_in_set(mod, expected)
    mod = InferType()(mod)
    mod = ManifestAlloc()(mod)

    # pylint: disable=line-too-long
    # def @main(%p0: Tensor[(5, 5), float32],
    #           %p1: Tensor[(5, 5), float32],
    #           %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x_0 = raf.op.cublas.matmul;
    #   let %x_1 = (%p0, %p1);
    #   let %x_2 = raf.op.vm.alloc_storage(int64(100), int64(128), int32(1), int32(0), str"float32") /* ty=float32 */;
    #   let %x_3 = raf.op.vm.alloc_tensor(%x_2, [5, 5], str"float32", [5, 5]) /* ty=Tensor[(5, 5), float32] */;
    #   let %x_4 = (%x_3,);
    #   let %x_5 = raf.op.vm.invoke_op(%x_0, %x_1, %x_4) /* ty=() */;
    #   let %x1 = %x_3;
    #   let %x_6 = fn (%p01: Tensor[(5, 5), float32], %p11: Tensor[(5, 5), float32], Primitive=1, Dialect="tvm") -> Tensor[(5, 5), float32] {
    #     %0 = raf.op.tvm.multiply(%p01, %p11) /* ty=Tensor[(5, 5), float32] */;
    #     raf.op.tvm.relu(%0) /* ty=Tensor[(5, 5), float32] */
    #   };
    #   let %x_7 = (%x1, %p2);
    #   let %x_8 = raf.op.vm.alloc_storage(int64(100), int64(128), int32(1), int32(0), str"float32") /* ty=float32 */;
    #   let %x_9 = raf.op.vm.alloc_tensor(%x_8, [5, 5], str"float32", [5, 5]) /* ty=Tensor[(5, 5), float32] */;
    #   let %x_10 = (%x_9,);
    #   let %x_11 = raf.op.vm.invoke_op(%x_6, %x_7, %x_10) /* ty=() */;
    #   let %x3 = %x_9;
    #   %x3
    # }
    # pylint: enable=line-too-long
    expected = {
        "n_3": {},
        "x_0": {"param_0", "param_1", "param_2"},
        "x_1": {"param_0", "param_1", "param_2"},
        "x_2": {"param_0", "param_1", "param_2"},
        "x_3": {"param_0", "param_1", "param_2", "t_0"},
        "x_4": {"param_0", "param_1", "param_2", "t_1"},
        "x_5": {"param_0", "param_1", "param_2", "t_1"},
        "x1": {"param_2", "t_1"},
        "x_6": {"param_2", "t_1"},
        "x_7": {"param_2", "t_1"},
        "x_8": {"param_2", "t_1"},
        "x_9": {"param_2", "t_1", "t_2"},
        "x_10": {"param_2", "t_1", "t_3"},
        "x_11": {"param_2", "t_1", "t_3"},
        "x3": {"t_3"},
        "n_4": {"t_3"},
    }
    verify_live_in_set(mod, expected)


if __name__ == "__main__":
    pytest.main([__file__])
