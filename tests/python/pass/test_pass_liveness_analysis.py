# pylint: disable=invalid-name, no-self-use, too-many-locals, unused-variable, protected-access
# pylint: disable=too-many-arguments
import pytest
import mnm
from mnm._lib import tvm, relay
from mnm.ir import ScopeBuilder
from mnm._ffi.pass_ import InferType, LivenessAnalysis, ManifestAlloc
from mnm.testing import randn

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
        print("IR:\n%s" % mnm.ir.AsText(mod))
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
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, param_0, param_1, param_2, param_3):
            t_0 = mnm.add(param_0, param_0) # a1
            t_1 = mnm.add(t_0, param_1) # a2
            t_2 = mnm.add(t_1, param_2) # a3
            t_3 = mnm.add(t_2, t_0) # a4
            t_4 = mnm.add(t_3, param_3) # a5
            return t_4 # n_1

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
        "n_1": {"t_4"}
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_multi_outs():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, param_0, param_1, param_2, param_3, param_4):
            t_0 = mnm.relu(param_0) # a1
            res = mnm.batch_norm_train(t_0, param_3, param_4, param_1, param_2, 0.1, 1e-5) # a2
            t_1 = res[0] # a3
            t_2 = res[1]
            t_3 = res[2]
            t_4 = mnm.relu(t_1) # a4
            t_5 = mnm.relu(t_4) # a5
            return t_5 # n_1

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
        "n_1": {"t_5"}
    }

    mod = model._internal(*args).mod
    verify_live_in_set(mod, expected)


def test_tuple_input():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, tup):
            x = tup[0] # a1
            y = tup[1] # a2
            t_0 = mnm.add(x, y) # a3
            return t_0 # n_1

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
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, tup):
            x = tup[0] # a1
            t_0 = mnm.add(x, x) # a2
            t_1 = mnm.concatenate(tup) # a3
            ret = (t_0, t_1) # a4
            return ret # n_1

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
    p0 = mnm.ir.var("p0", shape=(10, 10))
    a_1 = sb.let("a1", mnm.ir.op.relu(p0))
    a_2 = sb.let("a2", a_1)
    a_3 = sb.let("a3", mnm.ir.op.relu(a_2))
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


def test_reshape():
    shape = (10, 10)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            t_0 = mnm.relu(x)
            t_1 = mnm.reshape(t_0, (shape[0] * shape[1],))
            t_2 = mnm.relu(t_1)
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
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        null = mnm.ir.const(None)

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
        let6 = relay.Let(a6, mnm.ir.op.vm_invoke_op(a2, a4, a5), let7)
        let5 = relay.Let(a5, relay.Tuple((a1,)), let6)
        # Test both binded and non-binded constants
        let4 = relay.Let(a4, relay.Tuple((x, y, a3, null)), let5)
        let3 = relay.Let(a3, null, let4)
        let2 = relay.Let(a2, add_op, let3)
        let1 = relay.Let(a1, mnm.ir.op.vm_alloc_tensor(a0, [5, 5], "float32", [5, 5]), let2)
        let0 = relay.Let(a0, mnm.ir.op.vm_alloc_storage(100, 64, 1, 0), let1)
        # pylint: disable=line-too-long
        # fn (%x: Tensor[(5, 5), float32], %y: Tensor[(5, 5), float32]) {
        #   let %a0 = mnm.op.vm.alloc_storage(int64(100), int64(64), int64(1), int64(0), str"float32");
        #   let %a1 = mnm.op.vm.alloc_tensor(%a0, TupleValue([int64(5), int64(5)]), str"float32", TupleValue([int64(5), int64(5)]));
        #   let %a2 = mnm.op.add;
        #   let %a3 = nullptr;
        #   let %a4 = (%x, %y, %a3, nullptr);
        #   let %a5 = (%a1,);
        #   let %a6 = mnm.op.vm.invoke_op(%a2, %a4, %a5);
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
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, param_0, param_1, param_2):
            t_0 = mnm.add(param_0, param_0) # a1
            t_1 = mnm.add(t_0, param_1) # a2
            t_2 = mnm.add(t_1, param_2) # a3
            return t_2 # n_1

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
    #   let %x_0 = nullptr /* ty=() */;
    #   let %x_1 = nullptr /* ty=() */;
    #   let %x_2 = mnm.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_3 = mnm.op.vm.alloc_tensor(%x_2, [5, 5], str"float32", [5, 5]);
    #   let %x_4 = mnm.op.add;
    #   let %x_5 = (%param_0, %param_0, %x_0, %x_1);
    #   let %x_6 = (%x_3,);
    #   let %x_7 = mnm.op.vm.invoke_op(%x_4, %x_5, %x_6);
    #   let %a1 = %x_3;
    #   let %x_8 = nullptr /* ty=() */;
    #   let %x_9 = nullptr /* ty=() */;
    #   let %x_10 = mnm.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_11 = mnm.op.vm.alloc_tensor(%x_10, [5, 5], str"float32", [5, 5]);
    #   let %x_12 = mnm.op.add;
    #   let %x_13 = (%a1, %param_1, %x_8, %x_9);
    #   let %x_14 = (%x_11,);
    #   let %x_15 = mnm.op.vm.invoke_op(%x_12, %x_13, %x_14);
    #   let %a2 = %x_11;
    #   let %x_16 = nullptr /* ty=() */;
    #   let %x_17 = nullptr /* ty=() */;
    #   let %x_18 = mnm.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_19 = mnm.op.vm.alloc_tensor(%x_18, [5, 5], str"float32", [5, 5]);
    #   let %x_20 = mnm.op.add;
    #   let %x_21 = (%a2, %param_2, %x_16, %x_17);
    #   let %x_22 = (%x_19,);
    #   let %x_23 = mnm.op.vm.invoke_op(%x_20, %x_21, %x_22);
    #   let %a3 = %x_19;
    #   %a3
    # }
    # pylint: enable=line-too-long

    expected = {
        "x_2": {'param_0', 'param_1', 'param_2'},
        "x_3": {'param_0', 'param_1', 'param_2', 't_0'},
        "x_4": {'param_0', 'param_1', 't_1', 'param_2'},
        "x_5": {'param_0', 'param_1', 't_1', 'param_2'},
        "x_6": {'param_0', 'param_1', 't_1', 'param_2'},
        "x_7": {'param_0', 'param_1', 't_1', 'param_2'},
        "a1": {'param_1', 't_1', 'param_2'},
        "x_10": {'param_1', 't_1', 'param_2'},
        "x_11": {'param_1', 't_1', 'param_2', 't_2'},
        "x_12": {'param_1', 't_1', 'param_2', 't_3'},
        "x_13": {'param_1', 't_1', 'param_2', 't_3'},
        "x_14": {'param_1', 't_1', 'param_2', 't_3'},
        "x_15": {'param_1', 't_1', 'param_2', 't_3'},
        "a2": {'t_3', 'param_2'},
        "x_18": {'t_3', 'param_2'},
        "x_19": {'t_3', 'param_2', 't_4'},
        "x_20": {'param_2', 't_5', 't_3'},
        "x_21": {'param_2', 't_5', 't_3'},
        "x_22": {'param_2', 't_5', 't_3'},
        "x_23": {'param_2', 't_5', 't_3'},
        "a3": {'t_5'},
        "n_4": {'t_5'},
    }

    verify_live_in_set(mod, expected)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_fuse_closure():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, p0, p1, p2):
            t_0 = mnm.matmul(p0, p1)
            t_1 = mnm.multiply(t_0, p2)
            t_2 = mnm.relu(t_1)
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
    with mnm.device("cuda"):
        mod = mnm._ffi.pass_.ToGraphNormalForm()(mod)
        mod = mnm._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = mnm._ffi.pass_.FuseDialect()(mod)
        mod = mnm._ffi.pass_.FuseTVM()(mod)
        mod = mnm._ffi.pass_.ToANormalForm()(mod)
        mod = mnm._ffi.pass_.InlinePrimitives()(mod)
    # fn (%p0: Tensor[(5, 5), float32],
    #     %p1: Tensor[(5, 5), float32],
    #     %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x1 = mnm.op.cublas.matmul(%p0, %p1) /* ty=Tensor[(5, 5), float32] */;
    #   %1 = fn (%p01: Tensor[(5, 5), float32], %p11: Tensor[(5, 5), float32],
    #            Primitive=1, Dialect="tvm") -> Tensor[(5, 5), float32] {
    #     %0 = mnm.op.tvm.multiply(%p01, %p11);
    #     mnm.op.tvm.relu(%0)
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

    # def @main(%p0: Tensor[(5, 5), float32],
    #           %p1: Tensor[(5, 5), float32],
    #           %p2: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
    #   let %x_0 = mnm.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_1 = mnm.op.vm.alloc_tensor(%x_0, [5, 5], str"float32",[5, 5]);
    #   let %x_2 = mnm.op.cublas.matmul;
    #   let %x_3 = (%p0, %p1);
    #   let %x_4 = (%x_1,);
    #   let %x_5 = mnm.op.vm.invoke_op(%x_2, %x_3, %x_4);
    #   let %x1 = %x_1;
    #   let %x_6 = mnm.op.vm.alloc_storage(int64(100), int64(64), int32(1), int32(0), str"float32");
    #   let %x_7 = mnm.op.vm.alloc_tensor(%x_6, [5, 5], str"float32",[5, 5]);
    #   let %x_8 = fn (%p01: Tensor[(5, 5), float32],
    #                  %p11: Tensor[(5, 5), float32], Primitive=1, Dialect="tvm")
    #              -> Tensor[(5, 5), float32] {
    #     %0 = mnm.op.tvm.add(%p01, %p11, nullptr /* ty=() */, nullptr /* ty=() */);
    #     mnm.op.tvm.relu(%0)
    #   };
    #   let %x_9 = (%x1, %p2);
    #   let %x_10 = (%x_7,);
    #   let %x_11 = mnm.op.vm.invoke_op(%x_8, %x_9, %x_10);
    #   let %x3 = %x_7;
    #   %x3
    # }
    expected = {
        "n_3": {},
        "x_0": {"param_0", "param_1", "param_2"},
        "x_1": {"param_0", "param_1", "param_2", "t_0"},
        "x_2": {"param_0", "param_1", "param_2", "t_1"},
        "x_3": {"param_0", "param_1", "param_2", "t_1"},
        "x_4": {"param_0", "param_1", "param_2", "t_1"},
        "x_5": {"param_0", "param_1", "param_2", "t_1"},
        "x1": {"param_2", "t_1"},
        "x_6": {"param_2", "t_1"},
        "x_7": {"param_2", "t_1", "t_2"},
        "x_8": {"param_2", "t_1", "t_3"},
        "x_9": {"param_2", "t_1", "t_3"},
        "x_10": {"param_2", "t_1", "t_3"},
        "x_11": {"param_2", "t_1", "t_3"},
        "x3": {"t_3"},
        "n_4": {"t_3"},
    }
    verify_live_in_set(mod, expected)


if __name__ == "__main__":
    pytest.main([__file__])
