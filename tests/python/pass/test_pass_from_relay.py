# pylint: disable=protected-access, too-many-locals, attribute-defined-outside-init, no-self-use
# pylint: disable=too-many-arguments, missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring, too-many-lines, invalid-name
from operator import attrgetter

import pytest
import numpy as np
import mnm
from mnm.frontend import FrameworkModel
from mnm.testing import get_device_list, randint, randn, check, utils
from mnm.model.trace import _get_func_inputs
from mnm._ffi.pass_ import FromRelay, InferType
from mnm._core.module import IRModule
from mnm._core.executor import VMExecutor
from mnm._lib import tvm as _tvm
from mnm._lib import relay as _relay


def check_from_relay(m_model, r_func, args, check_model_structure=True,
                     check_correctness=True, device="cpu", disabled_pass=None):
    m_mod = m_model._internal(*args).mod
    m_func = m_mod["main"]
    ref_outs = m_model(*args)
    ref_outs = ref_outs if isinstance(ref_outs, (tuple, list)) else (ref_outs,)

    try:
        r_mod = _tvm.IRModule()
        r_mod["main"] = r_func
        disabled_pass = disabled_pass if disabled_pass is not None else []
        new_mod = FromRelay(disabled_pass)(r_mod)
        new_func = new_mod["main"]
    except Exception as err:  # pylint: disable=broad-except
        assert False, "Failed to convert the Relay function:\n%s\nReason:\n%s" % (
            str(r_func), str(err))

    if check_model_structure:
        assert _tvm.ir.structural_equal(
            m_func, new_func), "%s\nvs\n%s\n" % (str(m_func), str(new_func))

    assert InferType()(new_mod), "Type error of the model from Relay"

    new_model = FrameworkModel(new_mod, new_mod, {}, {})
    if device == "cuda":
        args = [arg.to(device=device) for arg in args]
        new_model.to(device=device)
    outs = new_model(*args)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    if check_correctness:
        assert len(ref_outs) == len(outs)
        for ref_out, out in zip(ref_outs, outs):
            check(ref_out, out)


def test_mnm_constant():
    data = mnm.array(1)

    def expected():
        a1 = _relay.var("a1")  # pylint: disable=invalid-name
        t_value = mnm._core.value.TensorValue.from_numpy(data.asnumpy())
        constant = mnm._ffi.ir._make.Constant(t_value)
        let = _relay.Let(a1, constant, a1)
        return _relay.Function([], let)

    r_func = _relay.Function([], _relay.const(1))
    r_mod = IRModule.from_expr(r_func)
    m_mod = FromRelay()(r_mod)
    assert _tvm.ir.structural_equal(m_mod["main"], expected())
    model = FrameworkModel(m_mod, m_mod, {}, {})
    check(data, model())


def test_mnm_module():
    f1 = _relay.GlobalVar("f1")  # pylint: disable=invalid-name

    def get_tvm_mod():
        x = _relay.var("x", shape=(1, 100))
        tanh = _relay.tanh(x)
        tvm_mod = _tvm.IRModule()
        tvm_mod[f1] = _relay.Function([x], tanh)
        tvm_mod = _relay.transform.InferType()(tvm_mod)

        y = _relay.var("y", shape=(1, 100))
        out = f1(y)
        main = _relay.GlobalVar("main")
        tvm_mod[main] = _relay.Function([y], out)
        tvm_mod = _relay.transform.InferType()(tvm_mod)
        return tvm_mod

    def expected():
        a1 = _relay.var("a1")  # pylint: disable=invalid-name
        x = _relay.var("x", shape=(1, 100))
        tanh_op = mnm._ffi.op.GetOp("mnm.op.tanh")
        let = _relay.Let(a1, _relay.Call(tanh_op, [x]), a1)
        f1_out = _relay.Function([x], let)
        mod = IRModule({f1: f1_out})

        a1 = _relay.var("a1")  # pylint: disable=invalid-name
        y = _relay.var("y", shape=(1, 100))
        let = _relay.Let(a1, _relay.Call(f1, [y]), a1)
        main_out = _relay.Function([y], let)
        mod[_relay.GlobalVar("main")] = main_out
        return mod

    tvm_mod = get_tvm_mod()
    m_mod = FromRelay()(tvm_mod)
    expected_mod = expected()
    assert _tvm.ir.structural_equal(m_mod["f1"], expected_mod["f1"])
    assert _tvm.ir.structural_equal(m_mod["main"], expected_mod["main"])

    model = FrameworkModel(m_mod, m_mod, {}, {})

    m_x, n_x = randn((1, 100))

    # Check that VM can execute multi-module functions
    vm_executor, args = utils.get_vm_executor(model, 'cpu', [m_x], mnm._ffi.pass_.FuseOps(1))
    m_out = vm_executor(*args)
    ref_out = np.tanh(n_x)
    check(m_out, ref_out)


@pytest.mark.parametrize("op_name", [
    "copy", "abs", "ceil", "floor", "log", "exp", "cos", "sin", "sign", "round",
    "relu", "erf", "sqrt", "rsqrt", "atan", "negative", "sigmoid", "tanh", "batch_flatten"
])
@pytest.mark.parametrize("shape", [(2, 2)])
def test_mnm_unary_op(op_name, shape):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return attrgetter(op_name)(mnm)(x)

    model = TestModel()
    m_x, _ = randn(shape)

    # relay ir
    r_x = _relay.var("x", shape=shape)
    if op_name in ["relu", "batch_flatten"]:
        r_func = _relay.Function(params=[r_x], body=attrgetter(
            "nn.%s" % op_name)(_relay)(r_x))
    else:
        r_func = _relay.Function(
            params=[r_x], body=attrgetter(op_name)(_relay)(r_x))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("op_name", [
    "add", "subtract", "divide", "multiply", "power",
    "greater", "maximum", "minimum"
])
@pytest.mark.parametrize("shape", [(2, 2)])
def test_mnm_binary_tensor_op(op_name, shape):
    # meta ir
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return attrgetter(op_name)(mnm)(x, y)

    model = TestModel()
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)

    # relay ir
    r_x = _relay.var("x", shape=shape)
    r_y = _relay.var("y", shape=shape)
    r_func = _relay.Function(
        params=[r_x, r_y], body=attrgetter(op_name)(_relay)(r_x, r_y))

    check_from_relay(model, r_func, [m_x, m_y])


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("shape", [(3, 4, 2, 2)])
@pytest.mark.parametrize("repeats", [2])
@pytest.mark.parametrize("axis", [-1])
def test_repeat(shape, repeats, axis, dtype):

    class Repeat(mnm.Model):
        def build(self, repeats, axis=0):
            self._repeats = repeats
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.repeat(x, repeats=self._repeats, axis=self._axis)

    model = Repeat(repeats, axis)
    m_x, _ = randn(shape, dtype=dtype)

    r_x = _relay.var("x", shape=shape)
    r_func = _relay.Function(
        params=[r_x], body=_relay.repeat(r_x, repeats, axis))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [[(5, 4, 3), (1, 2)]])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("mode", ["clip", "wrap"])
def test_take(shape, axis, dtype, mode):
    # pylint:disable=import-outside-toplevel
    from functools import reduce
    import operator

    class Take(mnm.Model):
        def build(self, axis, mode):
            self._axis = axis
            self._mode = mode

        @mnm.model.trace
        def forward(self, x, indices):
            return mnm.take(x, indices=indices, axis=self._axis, mode=self._mode)

    size = reduce(operator.mul, shape[0],
                  1) if axis is None else shape[0][axis]
    m_x, _ = randn(shape[0], dtype=dtype)
    m_indices, _ = randint(shape[1], low=0, high=size)
    model = Take(axis, mode)

    r_x = _relay.var("x", shape=shape[0])
    r_indices = _relay.var("i", shape=shape[1], dtype="int64")
    r_func = _relay.Function(params=[r_x, r_indices], body=_relay.take(
        r_x, r_indices, axis, mode=mode))

    check_from_relay(model, r_func, [m_x, m_indices])


@pytest.mark.parametrize("max_length", [5])
@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("other_feature_dims", [[3, 4]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_sequence_mask(max_length, batch_size, other_feature_dims,
                       axis, dtype):
    class SequenceMask(mnm.Model):
        def build(self, mask_value, axis=0):
            self._axis = axis
            self._mask_value = mask_value

        @mnm.model.trace
        def forward(self, x, sequence_length):
            return mnm.sequence_mask(x, sequence_length,
                                     mask_value=self._mask_value, axis=self._axis)

    x_shape = [max_length, batch_size] if axis == 0 else [
        batch_size, max_length]
    x_shape += other_feature_dims
    model = SequenceMask(-10, axis)

    m_x, _ = randn(x_shape, dtype=dtype)
    m_length, _ = randint([batch_size], low=0, high=max_length, dtype=dtype)

    r_x = _relay.var("x", shape=x_shape)
    r_l = _relay.var("l", shape=[batch_size])
    r_func = _relay.Function(
        params=[r_x, r_l], body=_relay.sequence_mask(r_x, r_l, -10, axis))

    check_from_relay(model, r_func, [m_x, m_length])


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("shape", [[10, 10, 10], [6, 8, 9, 10]])
@pytest.mark.parametrize("axis", [0, 1])
def test_reverse(shape, axis, dtype):

    class Reverse(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.reverse(x, self._axis)

    m_x, _ = randn(shape, dtype=dtype)
    model = Reverse(axis=axis)

    r_x = _relay.var("x", shape=shape)
    r_func = _relay.Function(params=[r_x], body=_relay.reverse(r_x, axis))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("inputs", [{"shape": (5, 5, 5), "seq_length": [1, 2, 3, 4, 5]}])
@pytest.mark.parametrize("axes", [[0, 1]])
def test_reverse_sequence(inputs, axes, dtype):

    class ReverseSequence(mnm.Model):
        def build(self, seq_axis, batch_axis):
            self._seq_axis = seq_axis
            self._batch_axis = batch_axis

        @mnm.model.trace
        def forward(self, x, seq_length):
            return mnm.reverse_sequence(x, seq_length, self._seq_axis, self._batch_axis)

    shape = inputs["shape"]
    m_seq_length = mnm.array(inputs["seq_length"], dtype="int64")
    seq_axis = axes[0]
    batch_axis = axes[1]
    m_x, _ = randn(shape, dtype=dtype)
    model = ReverseSequence(seq_axis, batch_axis)

    r_x = _relay.var("x", shape=shape)
    r_l = _relay.var("l", shape=(len(inputs["seq_length"]),), dtype="int64")
    r_func = _relay.Function(
        params=[r_x, r_l], body=_relay.reverse_sequence(r_x, r_l, seq_axis, batch_axis))

    check_from_relay(model, r_func, [m_x, m_seq_length])


@pytest.mark.parametrize("shape", [[[1, 4, 1], [1, 2, 4, 1]]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_broadcast_to(shape, dtype):

    class BroadcastTo(mnm.Model):
        def build(self, shape=None):
            self._shape = shape

        @mnm.model.trace
        def forward(self, x):
            return mnm.broadcast_to(x, self._shape)

    model = BroadcastTo(shape[1])
    m_x, _ = randn(shape[0], dtype=dtype)

    r_x = _relay.var("x", shape=shape[0])
    r_func = _relay.Function(
        params=[r_x], body=_relay.broadcast_to(r_x, shape[1]))

    check_from_relay(model, r_func, [m_x])

@pytest.mark.xfail(reason="broadcast_to_like with static shape will be simplified to broadcast_to")
@pytest.mark.parametrize("shape", [[[1, 4, 1], [1, 2, 4, 1]]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_broadcast_to_like(shape, dtype):

    class BroadcastToLike(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, broadcast_type):  # pylint: disable=no-self-use
            return mnm.broadcast_to_like(x, broadcast_type)

    model = BroadcastToLike()
    m_x, _ = randn(shape[0], dtype=dtype)
    broadcast_type, _ = randn(shape[1], dtype=dtype)

    r_x = _relay.var("x", shape=shape[0])
    r_b = _relay.var("b", shape=shape[1])
    r_func = _relay.Function(
        params=[r_x, r_b], body=_relay.broadcast_to_like(r_x, r_b))

    check_from_relay(model, r_func, [m_x, broadcast_type])


@pytest.mark.parametrize("shape", [[(2, 2), (1, 0)], [(2, 2), None]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_transpose(shape, dtype):

    class Transpose(mnm.Model):
        def build(self, axes=None):
            self._axes = axes

        @mnm.model.trace
        def forward(self, x):
            ret = mnm.transpose(x, self._axes)
            return ret

    axes = shape[1]
    model = Transpose(axes)
    m_x, _ = randn(shape[0], dtype=dtype)

    r_x = _relay.var("x", shape=shape[0])
    r_func = _relay.Function(params=[r_x], body=_relay.transpose(r_x, axes))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [[10, 20, 30]])
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("indices_or_sections", [(2, 4), 2, (2,)])
@pytest.mark.parametrize("dtype", ["float32"])
def test_split(shape, axis, indices_or_sections, dtype):

    class Split(mnm.Model):
        def build(self, indices_or_sections, axis):
            self._indices_or_sections = indices_or_sections
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            ret = mnm.split(x, self._indices_or_sections, self._axis)
            return ret

    model = Split(indices_or_sections, axis)
    m_x, _ = randn(shape, dtype=dtype)

    r_x = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(params=[r_x], body=_relay.split(
        r_x, indices_or_sections, axis).astuple())

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shapes", [[[2, 2, 2], [2, 3, 2], [2, 4, 2]]])
@pytest.mark.parametrize("axis", [-2])
@pytest.mark.parametrize("dtype", ["float32"])
def test_concatenate(shapes, axis, dtype):

    class Concatenate(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.concatenate([a, b, c], axis=self._axis)

    args = [randn(shape, dtype=dtype)[0] for shape in shapes]
    model = Concatenate(axis)

    r_vars = [_relay.var("x%d" % idx, shape=shape)
              for idx, shape in enumerate(shapes)]
    r_func = _relay.Function(
        params=r_vars, body=_relay.concatenate(r_vars, axis=axis))

    check_from_relay(model, r_func, args)


@pytest.mark.parametrize("shapes", [[[2, 2, 2], [2, 2, 2], [2, 2, 2]]])
@pytest.mark.parametrize("axis", [-1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_stack(shapes, axis, dtype):

    class Stack(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.stack([a, b, c], axis=self._axis)

    args = [randn(shape, dtype=dtype)[0] for shape in shapes]
    model = Stack(axis)

    r_vars = [_relay.var("x%d" % idx, shape=shape, dtype=dtype)
              for idx, shape in enumerate(shapes)]
    r_func = _relay.Function(
        params=r_vars, body=_relay.stack(r_vars, axis=axis))

    check_from_relay(model, r_func, args)


@pytest.mark.parametrize("shape", [[1, 4, 1, 2]])
@pytest.mark.parametrize("axis", [[0], None])
@pytest.mark.parametrize("dtype", ["float32"])
def test_squeeze(shape, axis, dtype):

    class Squeeze(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.squeeze(x, self._axis)

    model = Squeeze(axis)
    m_x, _ = randn(shape, dtype=dtype)

    r_var = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(params=[r_var], body=_relay.squeeze(r_var, axis))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [(2, 4, 1, 3), (1, 2, 3)])
@pytest.mark.parametrize("a_min", [0.3])
@pytest.mark.parametrize("a_max", [0.7])
@pytest.mark.parametrize("dtype", ["float32"])
def test_clip(shape, a_min, a_max, dtype):

    class Clip(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.clip(x, a_min, a_max)

    model = Clip()

    m_x, _ = randn(shape, dtype=dtype)

    r_var = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(
        params=[r_var], body=_relay.clip(r_var, a_min, a_max))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [(1, 2)])
@pytest.mark.parametrize("itype", ["float32"])
@pytest.mark.parametrize("otype", ["float16"])
def test_cast_n_cast_like(shape, itype, otype):

    class Cast(mnm.Model):
        def build(self, otype=None):
            self._otype = otype

        @mnm.model.trace
        def forward(self, data):
            return mnm.cast(data, self._otype)

    class CastLike(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data, dtype_like):
            return mnm.cast_like(data, dtype_like)

    m_x, _ = randn(shape, dtype=itype)
    cast_model = Cast(otype)
    r_var = _relay.var("x", shape=shape, dtype=itype)
    r_func = _relay.Function(params=[r_var], body=_relay.cast(r_var, otype))
    check_from_relay(cast_model, r_func, [m_x])

    m_x_like, _ = randn(shape, dtype=otype)
    cast_like_model = CastLike()
    cast_like_model(m_x, m_x_like)
    r_var = _relay.var("x", shape=shape, dtype=itype)
    r_var2 = _relay.var("y", shape=shape, dtype=otype)
    r_func = _relay.Function(
        params=[r_var, r_var2], body=_relay.cast_like(r_var, r_var2))
    check_from_relay(cast_like_model, r_func, [m_x, m_x_like])


@pytest.mark.parametrize("dshape", [[10, 11, 12, 13]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_gather(dshape, axis, dtype):
    class Gather(mnm.Model):
        def build(self, axis):
            self.axis = axis

        @mnm.model.trace
        def forward(self, data, indices):
            return mnm.gather(data, self.axis, indices)

    m_x, _ = randn(dshape, dtype=dtype)
    m_i, _ = randint(dshape, high=dshape[axis])
    model = Gather(axis)

    r_x = _relay.var("x", shape=dshape, dtype=dtype)
    r_i = _relay.var("i", shape=dshape, dtype="int64")
    r_func = _relay.Function(
        params=[r_x, r_i], body=_relay.gather(r_x, axis, r_i))

    check_from_relay(model, r_func, [m_x, m_i])


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("dshape", [[10, 11, 12]])
@pytest.mark.parametrize("ishape", [[3, 2]])
def test_gather_nd(dshape, dtype, ishape):

    class GatherNd(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data, indices):
            return mnm.gather_nd(data, indices)

    m_x, _ = randn(dshape, dtype=dtype)
    m_i, _ = randint(ishape, high=dshape[0: ishape[-1]], dtype="int64")
    model = GatherNd()

    r_x = _relay.var("x", shape=dshape, dtype=dtype)
    r_i = _relay.var("i", shape=ishape, dtype="int64")
    r_func = _relay.Function(
        params=[r_x, r_i], body=_relay.gather_nd(r_x, r_i))

    check_from_relay(model, r_func, [m_x, m_i])


@pytest.mark.parametrize("params", [
    {"orig_shape": (8, 8, 8, 8), "to_shape": (2, 2048)},
    {"orig_shape": (3, 3, 3, 3), "to_shape": (0, -1)},
])
@pytest.mark.parametrize("reverse", [False])
@pytest.mark.parametrize("dtype", ["float32"])
def test_reshape(params, reverse, dtype):

    class Reshape(mnm.Model):
        def build(self, shape, reverse=False):
            self._shape = shape
            self._reverse = reverse

        @mnm.model.trace
        def forward(self, x):
            return mnm.reshape(x, shape=self._shape, reverse=self._reverse)

    orig_shape, to_shape = params["orig_shape"], params["to_shape"]
    model = Reshape(shape=to_shape, reverse=reverse)
    m_x, _ = randn(orig_shape, dtype=dtype)

    r_x = _relay.var("x", shape=orig_shape, dtype=dtype)
    r_func = _relay.Function(params=[r_x], body=_relay.reshape(r_x, to_shape))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [[10, 3, 2, 5], [9, 12, 18, 2, 1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("num_newaxis", [5])
def test_expand_dims(shape, dtype, axis, num_newaxis):

    class ExpandDims(mnm.Model):
        def build(self, axis, num_newaxis):
            self.axis = axis
            self.num_newaxis = num_newaxis

        @mnm.model.trace
        def forward(self, x):
            return mnm.expand_dims(x, axis=self.axis, num_newaxis=self.num_newaxis)

    m_x, _ = randn(shape, dtype=dtype)
    model = ExpandDims(axis, num_newaxis)

    r_x = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(
        params=[r_x], body=_relay.expand_dims(r_x, axis, num_newaxis))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.xfail
@pytest.mark.parametrize("shape", [[3, 2]])
@pytest.mark.parametrize("val", [1])
def test_full(shape, val):
    class Full(mnm.Model):
        def build(self, shape, val):
            self.shape = shape
            self.val = val

        @mnm.model.trace
        def forward(self):
            return mnm.full(self.val, self.shape, "int64")

    model = Full(shape, val)

    r_func = _relay.Function(params=[],
                             body=_relay.full(_relay.const(val), shape=shape, dtype="int64"))
    check_from_relay(model, r_func, [], check_model_structure=False,
                     disabled_pass=["FoldConstant"])


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("params", [((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2])])
def test_strided_slice(dtype, params):
    class StridedSlice(mnm.Model):
        def build(self, begin, end, strides):
            self.begin = begin
            self.end = end
            self.strides = strides

        @mnm.model.trace
        def forward(self, data):
            return mnm.strided_slice(data, self.begin, self.end, self.strides)

    shape, begin, end, strides = params
    model = StridedSlice(begin, end, strides)
    m_x, _ = randn(shape, dtype=dtype)

    r_x = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(
        params=[r_x], body=_relay.strided_slice(r_x, begin, end, strides))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [(1, 4, 1), (3, 4, 2, 2)])
def test_where(shape):
    class Where(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, condition, x, y):
            return mnm.where(condition, x, y)

    model = Where()
    m_cond, _ = randint(shape, low=0, high=1, dtype="bool")
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)

    r_c = _relay.var("c", shape=shape, dtype="bool")
    r_x = _relay.var("x", shape=shape)
    r_y = _relay.var("y", shape=shape)
    r_func = _relay.Function(
        params=[r_c, r_x, r_y], body=_relay.where(r_c, r_x, r_y))

    check_from_relay(model, r_func, [m_cond, m_x, m_y])


@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_mnm_conv2d(xshape, wshape, stride, dilation, padding):
    # meta ir
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, w):
            return mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = TestModel()
    m_x, _ = randn(xshape)
    m_w, _ = randn(wshape)

    # relay ir
    r_x = _relay.var("x", shape=xshape)
    r_w = _relay.var("w", shape=wshape)
    r_c = _relay.nn.conv2d(r_x, r_w, strides=stride,
                           dilation=dilation, padding=padding)
    r_func = _relay.Function(params=[r_x, r_w], body=r_c)

    check_from_relay(model, r_func, [m_x, m_w])



@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(3, 16, 3, 3)])
@pytest.mark.parametrize("stride_output_padding", [
    (1, 0),
    (2, 1),
    (2, 0),
    (3, 2)
    ])
@pytest.mark.parametrize("dilation", [(1, 1)])
@pytest.mark.parametrize("padding", [0, 1])
def test_mnm_conv2d_trans(xshape, wshape, stride_output_padding, dilation, padding):
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, w):
            return mnm.conv2d_transpose(x, w, stride=stride, padding=padding,
                                        output_padding=output_padding,
                                        dilation=dilation,
                                        groups=1)
    model = TestModel()
    stride, output_padding = stride_output_padding
    stride = (stride, stride)
    output_padding = (output_padding, output_padding)
    m_x, _ = randn(xshape)
    m_w, _ = randn(wshape)
    # relay ir
    r_x = _relay.var("x", shape=xshape)
    r_w = _relay.var("w", shape=wshape)
    r_c = _relay.nn.conv2d_transpose(r_x, r_w, strides=stride, dilation=dilation,
                                     padding=padding,
                                     output_padding=output_padding)
    r_func = _relay.Function(params=[r_x, r_w], body=r_c)
    check_from_relay(model, r_func, [m_x, m_w])

# FIXME(@XIAO-XIA): Re-enable once dropout/dropout_dx can be dispatched to CuDNN.
@pytest.mark.xfail
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("p", [0.3, 0.7])
def test_contrib_dropout(device, shape, p):
    # pylint: disable=invalid-name,no-member
    class ContribDropout(mnm.Model):
        def build(self, p):
            self.p = p

        @mnm.model.trace
        def forward(self, x):
            return mnm._contrib_dropout(x, self.p)[0]

    model = ContribDropout(p)
    m_x, _ = randn(shape)

    r_x = _relay.var("x", shape=shape)
    r_c = _relay.nn.dropout(r_x, rate=p)
    r_func = _relay.Function(params=[r_x], body=r_c)

    # We cannot check the correctness for now due to random generated mask.
    check_from_relay(model, r_func, [m_x], check_correctness=False, device=device)


@pytest.mark.parametrize("kernel", [1, 2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize(
    "funcs",
    [
        [mnm._op.sym.max_pool2d, _relay.nn.max_pool2d],
        [mnm._op.sym.avg_pool2d, _relay.nn.avg_pool2d],
    ])
def test_mnm_pool2d(kernel, stride, padding, funcs):
    mnm_fwd, relay_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm_fwd(x, kernel=kernel, stride=stride, padding=padding)

    model = TestModel()
    m_x, _ = randn([8, 3, 32, 32])

    # relay ir
    r_x = _relay.var("x", shape=[8, 3, 32, 32])
    r_c = relay_fwd(r_x, pool_size=kernel, strides=stride, padding=padding)
    r_func = _relay.Function(params=[r_x], body=r_c)

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("out_shape", [(1, 1), (4, 4)])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize(
    "funcs",
    [
        [mnm._op.sym.adaptive_max_pool2d, _relay.nn.adaptive_max_pool2d],
        [mnm._op.sym.adaptive_avg_pool2d, _relay.nn.adaptive_avg_pool2d],
    ])
def test_mnm_adaptive_pool2d(out_shape, layout, funcs):
    mnm_fwd, relay_fwd = funcs

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm_fwd(x, shape=out_shape, layout=layout)

    model = TestModel()
    m_x, _ = randn([8, 3, 32, 32])

    # relay ir
    r_x = _relay.var("x", shape=[8, 3, 32, 32])
    r_c = relay_fwd(r_x, out_shape, layout)
    r_func = _relay.Function(params=[r_x], body=r_c)

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("shape", [(5, 4, 6, 9)])
@pytest.mark.parametrize("axis", [2])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("dtype", ["float32"])
def test_mnm_layer_norm(shape, axis, eps, dtype):
    class LayerNorm(mnm.Model):
        def build(self, axis, eps):
            self._axis = axis
            self._eps = eps

        @mnm.model.trace
        def forward(self, x, w, b):
            return mnm.layer_norm(x, w, b, axis=self._axis, eps=self._eps)

    model = LayerNorm(axis, eps)
    m_x, _ = randn(shape, dtype=dtype)
    m_w, _ = randn([shape[axis]], dtype=dtype)
    m_b, _ = randn([shape[axis]], dtype=dtype)

    r_x = _relay.var("x", shape=shape, dtype=dtype)
    r_w = _relay.var("w", shape=[shape[axis]], dtype=dtype)
    r_b = _relay.var("b", shape=[shape[axis]], dtype=dtype)
    r_func = _relay.Function(
        params=[r_x, r_w, r_b], body=_relay.nn.layer_norm(r_x, r_w, r_b, axis, eps))

    check_from_relay(model, r_func, [m_x, m_w, m_b])


@pytest.mark.parametrize("shape", [[32, 3, 224, 224]])
def test_mnm_batch_norm_train(shape):
    momentum = 0.1
    eps = 1e-5
    stats_shape = [shape[1]]
    m_x, _ = randn(shape)
    m_m, _ = randn(stats_shape)
    m_v, _ = randn(stats_shape, positive=True)
    m_w, _ = randn(stats_shape)
    m_b, _ = randn(stats_shape)

    class BatchNorm(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_x, m_w, m_b, m_m, m_v):
            res = mnm.batch_norm_train(m_x, m_m, m_v, m_w, m_b, momentum, eps)
            y = res[0]
            new_m = res[1]
            new_v = res[2]
            return (y, new_m, new_v)

    model = BatchNorm()

    r_x = _relay.var("x", shape=shape)
    r_w = _relay.var("w", shape=stats_shape)
    r_b = _relay.var("b", shape=stats_shape)
    r_m = _relay.var("m", shape=stats_shape)
    r_v = _relay.var("v", shape=stats_shape)
    r_func = _relay.Function(params=[r_x, r_w, r_b, r_m, r_v], body=_relay.nn.batch_norm(
        r_x, r_w, r_b, r_m, r_v, epsilon=eps)[0])

    check_from_relay(model, r_func, [m_x, m_m, m_v, m_w, m_b])


@pytest.mark.xfail
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("dimension", [((2, 3), ((1, 1), (2, 2)))])
@pytest.mark.parametrize("pad_value", [2])
@pytest.mark.parametrize("pad_mode", ["constant"])
def test_pad(dtype, dimension, pad_value, pad_mode):
    shape, pad_width = dimension
    mnm_pad_width = []
    for width in pad_width:
        mnm_pad_width += width

    class Pad(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, m_x):
            return mnm.pad(m_x, mnm_pad_width, pad_value, pad_mode)

    m_x, _ = randn(shape, dtype=dtype)
    model = Pad()

    r_x = _relay.var("x", shape=shape, dtype=dtype)
    r_func = _relay.Function(params=[r_x], body=_relay.nn.pad(
        r_x, pad_width, pad_value, pad_mode))

    check_from_relay(model, r_func, [m_x])

@pytest.mark.parametrize("trans", [[False, False], [False, True], [True, False], [True, True]])
def test_dense_pattern(trans):

    class TransposeDense(mnm.Model):
        def build(self, trans):
            self.op_name = "matmul"
            if trans[0] or not trans[1]:
                self.op_name += "_"
                self.op_name += "t" if trans[0] else "n"
                # dense is matmul_nt so the second input is already transposed
                self.op_name += "n" if trans[1] else "t"

        @mnm.model.trace
        def forward(self, m_x, m_y):
            x = mnm.relu(m_x)
            return getattr(mnm, self.op_name)(x, m_y)

    model = TransposeDense(trans)
    m_x, _ = randn((10, 10), dtype="float32")
    m_y, _ = randn((10, 10), dtype="float32")

    r_x = _relay.var("x", shape=(10, 10), dtype="float32")
    r_y = _relay.var("x", shape=(10, 10), dtype="float32")
    tran_x = _relay.nn.relu(r_x)
    tran_x = _relay.transpose(tran_x) if trans[0] else tran_x
    tran_y = _relay.transpose(r_y) if trans[1] else r_y
    r_out = _relay.nn.dense(tran_x, tran_y)
    r_func = _relay.Function(params=[r_x, r_y], body=r_out)

    check_from_relay(model, r_func, [m_x, m_y])

@pytest.mark.parametrize("shape", [(4, 3, 4, 5)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("keep", [0, 1])
def test_sum(shape, axis, keep):
    class Sum(mnm.Model):
        def build(self, axis, keep):
            self.axis = axis
            self.keep = keep

        @mnm.model.trace
        def forward(self, x):
            return mnm.sum(x, self.axis, self.keep)
    model = Sum(axis, keep)
    m_x, _ = randn(shape)
    r_x = _relay.var("x", shape=shape)
    r_func = _relay.Function(params=[r_x], body=_relay.sum(r_x, axis, keep))

    check_from_relay(model, r_func, [m_x])


@pytest.mark.parametrize("data", [[1, 10, 2], [1, 10, 3]])
def test_arange(data):
    start, stop, step = data
    class Arange(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, start, stop, step):
            return mnm.arange(start, stop, step)
    model = Arange()
    dtype = "float32"
    m_start = mnm.array(start, dtype=dtype)
    m_stop = mnm.array(stop, dtype=dtype)
    m_step = mnm.array(step, dtype=dtype)
    r_start = _relay.var("start", shape=[], dtype=dtype)
    r_stop = _relay.var("stop", shape=[], dtype=dtype)
    r_step = _relay.var("step", shape=[], dtype=dtype)
    x = _relay.arange(r_start, r_stop, r_step)
    r_func = _relay.Function([r_start, r_stop, r_step], x)
    check_from_relay(model, r_func, [m_start, m_stop, m_step])


@pytest.mark.parametrize("data_shape, index_shapes", [
    ((10, 5), [(3, 4), (3, 1)]),
    ((10, 5, 4), [(1, 2, 3), (1, 2, 3)])
])
def test_adv_index(data_shape, index_shapes):
    class Index(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, index0, index1):
            return mnm.adv_index([x, index0, index1])

    dtype = "float32"
    m_x, _ = randn(data_shape)
    m_indices = []
    model = Index()
    inputs = [_relay.var("data", _relay.TensorType(data_shape, dtype))]
    for i, index_shape in enumerate(index_shapes):
        limit = data_shape[i]
        index = np.random.uniform(0, limit - 1, size=index_shape).astype("int64")
        m_indices.append(mnm.array(index))
        inputs.append(_relay.var("index_{}".format(i), _relay.TensorType(index_shape, "int64")))

    out = _relay.op.adv_index(inputs)
    r_func = _relay.Function(inputs, out)

    check_from_relay(model, r_func, [m_x, m_indices[0], m_indices[1]])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_full_fusion(dtype):
    r_c = _relay.const(2, dtype)
    out = _relay.full(r_c, shape=(5, 5), dtype=dtype)
    out = _relay.multiply(_relay.const(1.0, dtype), out)
    r_func = _relay.Function(params=[], body=out)
    r_mod = _tvm.IRModule()
    r_mod["main"] = r_func
    mod = FromRelay()(r_mod)
    m = FrameworkModel(mod, mod, {}, {})

    record = m._internal()
    mod = record.mod
    mod = mnm._ffi.pass_.FuseOps(1)(mod)
    vm_inputs = _get_func_inputs(record, [], {}, get_handle=False)
    vm = VMExecutor(mod, "llvm")
    vm_exec = vm.make_executor()
    out = vm_exec(*vm_inputs)
    ref = np.ones((5, 5), dtype=dtype) * 2
    assert out.dtype == dtype
    check(ref, out)


@pytest.mark.parametrize("shape", [(100, ), (4, 4), (16, 3)])
def test_threefry_generate(shape):
    class ThreefryGenerate(mnm.Model):
        def build(self, shape):
            self.shape = shape

        @mnm.model.trace
        def forward(self, key):
            return mnm.threefry_generate(key, self.shape)

    m_key = mnm.array(_relay.random.threefry_key(0).data.asnumpy(), dtype="uint64")
    model = ThreefryGenerate(shape)

    r_key = _relay.var("key", shape=(10,), dtype="uint64")
    r_func = _relay.Function(params=[r_key], body=_relay.random.threefry_generate(
        r_key, shape))

    check_from_relay(model, r_func, [m_key])


def test_threefry_split():
    class ThreefrySplit(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, key):
            return mnm.threefry_split(key)

    m_key = mnm.array(_relay.random.threefry_key(0).data.asnumpy(), dtype="uint64")
    model = ThreefrySplit()

    r_key = _relay.var("key", shape=(10,), dtype="uint64")
    r_func = _relay.Function(params=[r_key], body=_relay.random.threefry_split(
        r_key))

    check_from_relay(model, r_func, [m_key])


@pytest.mark.parametrize("shape", [(4, 3, 4, 5)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("keep", [0, 1])
def test_mean(shape, axis, keep):
    class Mean(mnm.Model):
        def build(self, axis, keep):
            self.axis = axis
            self.keep = keep

        @mnm.model.trace
        def forward(self, x):
            return mnm.mean(x, self.axis, self.keep)
    model = Mean(axis, keep)
    m_x, _ = randn(shape)
    r_x = _relay.var("x", shape=shape)
    r_func = _relay.Function(params=[r_x], body=_relay.mean(r_x, axis, keep))

    check_from_relay(model, r_func, [m_x])


if __name__ == "__main__":
    pytest.main([__file__])
