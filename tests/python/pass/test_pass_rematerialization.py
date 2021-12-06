
# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
# pylint: disable=too-many-statements, no-self-use, too-many-arguments
import numpy as np
import pytest
import mnm
from mnm._core.device import Device
from mnm._core.executor import VMExecutor
from mnm._ffi.memory_pool import InitPool
from mnm.ir import ScopeBuilder
from mnm.model import Conv2d
from mnm.model.trace import _get_func_inputs
from mnm.testing import run_infer_type, randn

import tvm
from tvm import relay

def verify_remat(model_or_mod, args, budget_in_mbs, expected_ir, expected_peaks):
    """Verify the result of rematerialization pass.

    Parameters
    ----------
    model_or_mod: Union[IRModule, mnm.Model]
        The given model or module.
    args: List[mnm.ndarray]
        The input arguments.
    budget_in_mbs: int
        The user-specified budget in MBs.
    expected_ir: relay.Function
        The expected IR after rematerialization.
    expected_peaks: Tuple[float, float]
        The expected peak memory in MBs without and with rematerialization.
    """
    if not isinstance(model_or_mod, tvm.IRModule):
        record = model_or_mod._internal(*args)
        args = _get_func_inputs(record, args, {}, get_handle=False)
        mod = record.mod
    else:
        mod = model_or_mod

    ir_mod = mod
    with Device("cpu"):
        with mnm.ir.PassContext(config={"mnm.memory_budget": int(budget_in_mbs * 1048576)}):
            ir_mod = mnm._ffi.pass_.InferType()(ir_mod)
            ir_mod = mnm._ffi.pass_.InlinePrimitives()(ir_mod)
            ir_mod = mnm._ffi.pass_.InplaceUpdate()(ir_mod)
            try:
                ir_mod = mnm._ffi.pass_.Rematerialization()(ir_mod)
            except Exception as err: # pylint: disable=broad-except
                assert expected_ir is None, "Unexpected rematerialization failure: %s" % str(err)
                return

    if expected_ir is not None:
        expected_ir = run_infer_type(expected_ir)
        assert tvm.ir.structural_equal(expected_ir, ir_mod["main"]), \
            "\nExpected:\n%s\nGot\n%s" % (mnm.ir.AsText(expected_ir), mnm.ir.AsText(ir_mod["main"]))

    # Use CPU to avoid workspace memory.
    device = "cpu"
    param_size = sum([np.prod(arg.shape) * 4 / 1048576 for arg in args])
    for with_remat, expected_peak in zip([False, True], expected_peaks):
        InitPool(Device(device), "page_unit_pool")
        budget = budget_in_mbs if with_remat else 10000
        with tvm.transform.PassContext(opt_level=3,
                                       disabled_pass=["FuseTVM", "FuseDialect"],
                                       config={"mnm.memory_budget": int(budget * 1048576)}):
            mnm.utils.memory_profiler.reset()
            mnm.utils.memory_profiler.start()
            VMExecutor(mod, device).make_executor()(*args)
            mnm.utils.memory_profiler.stop()

        ret_map = mnm.utils.memory_profiler.get_max_memory_info(mnm.Device(device))
        peak_memory = ret_map["max_allocated"].value + param_size
        assert abs(expected_peak - peak_memory) < 0.1, \
            "Incorrect peak memory with remat=%s" % with_remat


@pytest.mark.parametrize("budget_type", ["low", "remat", "high"])
def test_simple(budget_type):
    shape = (16, 16, 64, 64) # 4 MBs
    data_size, weight_size = np.prod(shape), np.prod((16, 16, 3, 3))

    # Determine the budget.
    peak_memory = data_size * 6 + weight_size
    before_peak = peak_memory
    if budget_type == "high":
        budget = peak_memory
    elif budget_type == "remat":
        # Need to rematerialize to fit into this budget.
        budget = peak_memory - data_size
    else:
        # No way to fit into this budget.
        budget = peak_memory - data_size - 1

    # Make them MBs.
    to_mbs = lambda x: (4 * x) / 1048576
    budget = to_mbs(budget)
    before_peak = to_mbs(before_peak)

    class Model(mnm.Model):
        def build(self):
            self.conv = Conv2d(16, 16, kernel_size=(3, 3), padding=1, bias=False)

        @mnm.model.trace
        def forward(self, x):
            a_1 = self.conv(x)
            a_2 = mnm.softmax(a_1)
            a_3 = mnm.softmax(a_2)
            a_4 = mnm.softmax(a_3)

            a_5 = mnm.softmax_dx(a_3, a_4, a_4)
            a_6 = mnm.softmax_dx(a_2, a_3, a_5)
            a_7 = mnm.softmax_dx(a_1, a_2, a_6)
            a_8 = mnm.conv2d_dx(x, a_1, a_7, shape, 1, 1, 1, 1)
            a_9 = mnm.softmax(a_8)
            return a_9

    model = Model()
    m_x, _ = randn(shape, device="cpu")

    def expected():
        if budget_type == "low":
            return None

        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        conv2d_call = lambda x, w: relay.Call(conv2d_op,
                                              [x, w, mnm.ir.const([1]), mnm.ir.const([1]),
                                               mnm.ir.const([1]), mnm.ir.const(1),
                                               mnm.ir.const("NCHW"), mnm.ir.const("OIHW"),
                                               mnm.ir.const("NCHW")])
        conv2d_dx_op = mnm._ffi.op.GetOp("mnm.op.conv2d_dx")
        conv2d_dx_call = lambda x, y, dy: relay.Call(conv2d_dx_op, [x, y, dy,
                                                                    mnm.ir.const([16, 16, 64, 64]),
                                                                    mnm.ir.const([1]),
                                                                    mnm.ir.const([1]),
                                                                    mnm.ir.const([1]),
                                                                    mnm.ir.const(1)])
        softmax_op = mnm._ffi.op.GetOp("mnm.op.softmax")
        softmax_dx_op = mnm._ffi.op.GetOp("mnm.op.softmax_dx")
        minus_one = mnm.ir.const(-1)

        data = mnm.ir.var("x", shape=shape)
        weight = mnm.ir.var("w", shape=(16, 16, 3, 3))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", conv2d_call(data, weight))
        a_2 = sb.let("a2", relay.Call(softmax_op, [a_1, minus_one]))
        a_3 = sb.let("a3", relay.Call(softmax_op, [a_2, minus_one]))
        a_4 = sb.let("a4", relay.Call(softmax_op, [a_3, minus_one]))
        a_5 = sb.let("a5", relay.Call(softmax_dx_op, [a_3, a_4, a_4, minus_one]))

        # Rematerialize x_* tensors for a lower budget.
        if budget_type == "remat":
            x_0 = sb.let("x_1", relay.Call(softmax_op, [a_1, minus_one]))
        else:
            x_0 = a_2
        a_6 = sb.let("a6", relay.Call(softmax_dx_op, [x_0, a_3, a_5, minus_one]))

        # In the case os consecutive uses, we can reuse the rematerialized tensors.
        if budget_type == "remat":
            x_1 = sb.let("x_0", conv2d_call(data, weight))
        else:
            x_1 = a_1
        a_7 = sb.let("a7", relay.Call(softmax_dx_op, [x_1, x_0, a_6, minus_one]))

        # In the case os consecutive uses, we can reuse the rematerialized tensors.
        if budget_type == "remat":
            x_2 = x_1
        else:
            x_2 = a_1
        a_8 = sb.let("a8", conv2d_dx_call(data, x_2, a_7))
        a_9 = sb.let("a9", relay.Call(softmax_op, [a_8, minus_one]))

        sb.ret(a_9)
        return relay.Function([data, weight], sb.get())

    verify_remat(model, [m_x], budget, expected(), (before_peak, budget))

def test_closure():
    device = "cpu"
    shape = (16, 16, 64, 64) # 4 MBs

    def get_mod(with_remat=False):
        """This function includes a closure and has the peak memory 28.0088 MBs. We set
        the budget to 28 to enforce remating the tensor generated by the closure.
        """
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")
        conv2d_call = lambda x, w: relay.Call(conv2d_op,
                                              [x, w, mnm.ir.const([1]), mnm.ir.const([1]),
                                               mnm.ir.const([1]), mnm.ir.const(1),
                                               mnm.ir.const("NCHW"), mnm.ir.const("OIHW"),
                                               mnm.ir.const("NCHW")])
        conv2d_dx_op = mnm._ffi.op.GetOp("mnm.op.conv2d_dx")
        softmax_op = mnm._ffi.op.GetOp("mnm.op.softmax")
        softmax_dx_op = mnm._ffi.op.GetOp("mnm.op.softmax_dx")
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        null = mnm.ir.const(None)

        data = mnm.ir.var("x", shape=shape)
        weight = mnm.ir.var("w", shape=(16, 16, 3, 3))
        dy = mnm.ir.var("dy", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", conv2d_call(data, weight))
        a_2 = sb.let("a2", relay.Call(softmax_op, [a_1]))
        a_3 = sb.let("a3", relay.Call(softmax_op, [a_2]))

        # Closure
        p_0 = mnm.ir.var("p0", shape=shape)
        out = relay.Call(mnm._ffi.op.GetOp("mnm.op.tvm.relu"), [p_0])
        out = relay.Call(mnm._ffi.op.GetOp("mnm.op.tvm.softmax"), [out])
        closure = relay.Function([p_0], out)
        closure = closure.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        closure = closure.with_attr("Dialect", "tvm")

        a_4 = sb.let("a4", relay.Call(closure, [a_3]))
        a_5 = sb.let("a5", relay.Call(softmax_dx_op, [a_2, a_3, a_4]))
        z_0 = sb.let("z0", relay.Call(softmax_dx_op, [a_2, a_3, a_5])) # Can only kill a_4.
        if with_remat:
            x_0 = sb.let("x_0", relay.Call(closure, [a_3]))
        else:
            x_0 = a_4
        a_6 = sb.let("a6", relay.Call(softmax_dx_op, [a_3, x_0, z_0]))
        a_7 = sb.let("a7", relay.Call(conv2d_dx_op, [data, weight, a_6,
                                                     mnm.ir.const([16, 16, 64, 64]),
                                                     mnm.ir.const([1]), mnm.ir.const([1]),
                                                     mnm.ir.const([1]), mnm.ir.const(1)]))
        # Since the memory consumption at a7 is within the budget, we do not force free
        # x_0 at a7, so we can reuse the rematerialized x_0 here.
        if with_remat:
            x_4 = x_0
        else:
            x_4 = a_4
        a_8 = sb.let("a8", relay.Call(add_op, [x_4, a_7, null, null]))
        a_9 = sb.let("a9", relay.Call(softmax_op, [a_8]))
        sb.ret(a_9)
        func = relay.Function([data, weight, dy], sb.get())
        return tvm.IRModule.from_expr(func)

    m_x, _ = randn(shape, device=device)
    m_w, _ = randn((16, 16, 3, 3), device=device)
    m_dy, _ = randn(shape, device=device)
    verify_remat(get_mod(), [m_x, m_w, m_dy], 28, get_mod(with_remat=True)["main"],
                 (28.0088, 24.0088))


@pytest.mark.parametrize("share", [False, True])
def test_inplace(share):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, param_0, param_1, param_2):
            a_1 = mnm.add(param_0, param_0)
            a_2 = mnm.add(a_1, param_1, out=a_1) if share else mnm.add(a_1, param_1)
            a_3 = mnm.add(a_2, param_2)
            a_4 = mnm.softmax(a_3)
            a_5 = mnm.softmax(a_4)
            a_6 = mnm.add(a_5, a_1, out=a_5)
            return a_6

    device = "cpu"
    shape = (512, 512) # 1 MB
    model = Model()
    m_a, _ = randn(shape, device=device)
    m_b, _ = randn(shape, device=device)
    m_c, _ = randn(shape, device=device)
    args = [m_a, m_b, m_c]

    def expected():
        """The expected result of non-sharing a_1 and a_2"""
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        softmax_op = mnm._ffi.op.GetOp("mnm.op.softmax")
        minus_one = mnm.ir.const(-1)
        null = mnm.ir.const(None)

        p_0 = mnm.ir.var("p0", shape=shape)
        p_1 = mnm.ir.var("p1", shape=shape)
        p_2 = mnm.ir.var("p2", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(add_op, [p_0, p_0, null, null]))
        a_2 = sb.let("a2", relay.Call(add_op, [a_1, p_1, null, null]))
        a_3 = sb.let("a3", relay.Call(add_op, [a_2, p_2, null, null]))
        a_4 = sb.let("a4", relay.Call(softmax_op, [a_3, minus_one]))
        a_5 = sb.let("a4", relay.Call(softmax_op, [a_4, minus_one]))
        x_0 = sb.let("x0", relay.Call(add_op, [p_0, p_0, null, null])) # remat
        a_6 = sb.let("a4", relay.Call(add_op, [a_5, x_0, a_5, null]))
        sb.ret(a_6)
        return relay.Function([p_0, p_1, p_2], sb.get())

    if share:
        # The only candidate cannot be rematerialized when sharing, so failed.
        verify_remat(model, args, 5, None, ())
    else:
        verify_remat(model, args, 5, expected(), (6, 5))


def test_tuple():
    class Model(mnm.Model):
        def build(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            self.batch_norm = mnm.model.nn.BatchNorm(num_features, eps, momentum, affine)

        @mnm.model.trace
        def forward(self, x):
            # Note that BatchNorm model will mutate IR. See the expected IR for details.
            a_1 = mnm.relu(x)
            a_2 = self.batch_norm(x)
            a_4 = mnm.add(a_1, a_2)
            a_5 = mnm.relu(a_4)
            a_6 = mnm.add(a_4, a_5)
            a_7 = mnm.add(a_2, a_6)
            a_8 = mnm.add(a_1, a_7)
            return a_8

    device = "cpu"
    shape = (512, 512) # 1 MB.
    stats_shape = [shape[1]]
    model = Model(stats_shape)
    m_x, _ = randn(shape, device=device, requires_grad=True)
    args = [m_x]

    def expected():
        """a6 has to free a 1 MB tensor and the candidates are a1 and a2. Since a2 (i.e., BN) is
        a tuple and two of its tensors are still alive, we should free a1 (i.e., ReLU) and
        remat later.
        """
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        bn_op = mnm._ffi.op.GetOp("mnm.op.batch_norm_train")
        momentum = mnm.ir.const(0.1)
        eps = mnm.ir.const(1e-5)
        null = mnm.ir.const(None)

        p_0 = mnm.ir.var("p0", shape=shape)
        bn_b = mnm.ir.var("bn_b", shape=stats_shape)
        bn_m = mnm.ir.var("bn_m", shape=stats_shape)
        bn_v = mnm.ir.var("bn_v", shape=stats_shape)
        bn_w = mnm.ir.var("bn_w", shape=stats_shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [p_0]))
        a_2 = sb.let("a2", relay.Call(bn_op, [p_0, bn_m, bn_v, bn_w, bn_b, momentum, eps]))
        x_0 = sb.let("x_0", relay.TupleGetItem(a_2, 0))
        a_4 = sb.let("a4", relay.Call(add_op, [a_1, x_0, null, null]))
        a_5 = sb.let("a5", relay.Call(relu_op, [a_4]))
        a_6 = sb.let("a6", relay.Call(add_op, [a_4, a_5, null, null]))
        x_1 = sb.let("x_1", relay.TupleGetItem(a_2, 0))
        a_7 = sb.let("a7", relay.Call(add_op, [x_1, a_6, null, null]))
        x_2 = sb.let("x_2", relay.Call(relu_op, [p_0])) # remat
        a_8 = sb.let("a8", relay.Call(add_op, [x_2, a_7, null, null]))
        a_9 = sb.let("a9", relay.TupleGetItem(a_2, 1), may_share=bn_m)
        a_10 = sb.let("a10", relay.TupleGetItem(a_2, 2), may_share=bn_v)
        a_11 = sb.let("a11", relay.Tuple([a_8, a_9, a_10]))
        sb.ret(a_11)
        return relay.Function([p_0, bn_b, bn_m, bn_v, bn_w], sb.get())

    stats_size_mb = (stats_shape[0] * 4) / 1048576
    # Inputs are 1 data + 4 stats tensors.
    param_size = 1 + 4 * stats_size_mb
    # 2 updated intermediate stats + 5 intermediate tensors.
    peak_size = param_size + 2 * stats_size_mb + 5
    verify_remat(model, args, peak_size - 1, expected(), (peak_size, peak_size - 1))


def test_reshape():
    device = "cpu"
    shape = (512, 512) # 1 MB.

    add_op = mnm._ffi.op.GetOp("mnm.op.add")
    relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
    reshape_op = mnm._ffi.op.GetOp("mnm.op.reshape")
    null = mnm.ir.const(None)
    new_shape = mnm.ir.const((262144,), dtype="int32")

    def get_mod():
        p_0 = mnm.ir.var("x", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [p_0]))
        a_2 = sb.let("a2", relay.Call(relu_op, [p_0]))
        a_3 = sb.let("a3", relay.Tuple([a_1, a_2]))
        a_4 = sb.let("a4", relay.TupleGetItem(a_3, 0))
        a_5 = sb.let("a5", relay.Call(reshape_op, [a_4, new_shape]))
        a_6 = sb.let("a6", relay.TupleGetItem(a_3, 1))
        a_7 = sb.let("a7", relay.Call(reshape_op, [a_6, new_shape]))
        a_8 = sb.let("a8", relay.Call(relu_op, [a_5]))
        a_9 = sb.let("a9", relay.Call(relu_op, [a_7]))
        a_10 = sb.let("a10", relay.Call(add_op, [a_8, a_9, null, null]))
        a_11 = sb.let("a11", relay.Call(add_op, [a_5, a_10, null, null]))
        sb.ret(a_11)
        return tvm.IRModule.from_expr(relay.Function([p_0], sb.get()))

    def expected():
        """Tuple and TupleGetItem are simplified when generating reshapes (x_0, x_1).
        Meanwhile, x_2, x_3 are rematerialized.
        """
        p_0 = mnm.ir.var("x", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [p_0]))
        a_2 = sb.let("a2", relay.Call(relu_op, [p_0]))
        x_0 = sb.let("x_0", relay.Call(reshape_op, [a_1, new_shape]))
        a_8 = sb.let("a8", relay.Call(relu_op, [x_0]))
        x_1 = sb.let("x_1", relay.Call(reshape_op, [a_2, new_shape]))
        a_9 = sb.let("a9", relay.Call(relu_op, [x_1]))
        a_10 = sb.let("a10", relay.Call(add_op, [a_8, a_9, null, null]))
        x_3 = sb.let("a1", relay.Call(relu_op, [p_0]))
        x_4 = sb.let("x_2", relay.Call(reshape_op, [x_3, new_shape]))
        a_11 = sb.let("a11", relay.Call(add_op, [x_4, a_10, null, null]))
        sb.ret(a_11)
        return relay.Function([p_0], sb.get())

    m_x, _ = randn(shape, device=device, requires_grad=True)
    args = [m_x]
    verify_remat(get_mod(), args, 4, expected(), (5, 4))

def test_not_call():
    device = "cpu"
    shape = (512, 512) # 1 MB.
    stats_shape = [shape[1]]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, bn_m, bn_v, bn_w, bn_b):
            a_1 = mnm.relu(x)
            a_2 = mnm.batch_norm_train(x, bn_m, bn_v, bn_w, bn_b, 0.1, 1e-5)
            a_3 = a_2[0]
            a_4 = mnm.add(a_1, a_3)
            a_5 = mnm.relu(x)
            a_6 = mnm.add(a_4, a_5)
            a_7 = mnm.add(a_3, a_6)
            return a_7

    model = Model()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_m, _ = randn(stats_shape, device=device, requires_grad=True)
    m_v, _ = randn(stats_shape, device=device, requires_grad=True)
    m_w, _ = randn(stats_shape, device=device, requires_grad=True)
    m_b, _ = randn(stats_shape, device=device, requires_grad=True)
    args = [m_x, m_m, m_v, m_w, m_b]

    def expected():
        """Need to remat TupleGetItem as well."""
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        bn_op = mnm._ffi.op.GetOp("mnm.op.batch_norm_train")
        momentum = mnm.ir.const(0.1)
        eps = mnm.ir.const(1e-5)
        null = mnm.ir.const(None)

        p_x = mnm.ir.var("x", shape=shape)
        p_m = mnm.ir.var("m", shape=stats_shape)
        p_v = mnm.ir.var("v", shape=stats_shape)
        p_w = mnm.ir.var("w", shape=stats_shape)
        p_b = mnm.ir.var("b", shape=stats_shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(relu_op, [p_x]))
        a_2 = sb.let("a2", relay.Call(bn_op, [p_x, p_m, p_v, p_w, p_b, momentum, eps]))
        x_0 = sb.let("x_0", relay.TupleGetItem(a_2, 0))
        a_4 = sb.let("a4", relay.Call(add_op, [a_1, x_0, null, null]))
        a_5 = sb.let("a5", relay.Call(relu_op, [p_x]))
        a_6 = sb.let("a6", relay.Call(add_op, [a_4, a_5, null, null]))

        x_1 = sb.let("x_1", relay.Call(bn_op, [p_x, p_m, p_v, p_w, p_b, momentum, eps]))
        x_2 = sb.let("x_2", relay.TupleGetItem(x_1, 0))
        a_7 = sb.let("a7", relay.Call(add_op, [x_2, a_6, null, null]))
        sb.ret(a_7)
        return relay.Function([p_x, p_m, p_v, p_w, p_b], sb.get())

    verify_remat(model, args, 4.01172, expected(), (5.01172, 4.01172))


if __name__ == "__main__":
    pytest.main([__file__])
