import numpy as np
import pytest
import torch
import mnm
from mnm._ffi.pass_ import AutoDiff
from mnm.testing import check_type, run_infer_type, randn, randn_torch
from tvm.relay import TensorType, FuncType, TupleType


def randint(shape, *, low=0, high=None, ctx="cpu", dtype="int64"):
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


# pylint: disable=too-many-locals, import-outside-toplevel, attribute-defined-outside-init
@pytest.mark.parametrize("shape", [
    [(5, 4, 3), (1, 2)],
    [(6, 5), (2, 2)],
    [(1, 1), (2, 2, 2)],
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_take(shape, axis, dtype):
    from functools import reduce
    import operator

    class Take(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x, indices):
            return mnm.take(x, indices=indices, axis=self._axis)

    size = reduce(operator.mul, shape[0], 1) if axis is None else shape[0][axis]
    m_x, n_x = randn(shape[0], dtype=dtype)
    m_indices, n_indices = randint(shape[1], low=0, high=size)
    model = Take(axis)
    # forward
    m_func = model.get_relay_func(m_x, m_indices)
    m_func = run_infer_type(m_func)
    n_y = np.take(n_x, n_indices, axis=axis, mode="clip")
    x_ty = TensorType(n_x.shape, dtype=dtype)
    indices_ty = TensorType(n_indices.shape, dtype=m_indices.dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, indices_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dy_ty = TensorType(n_y.shape, dtype=dtype)
    dx_ty = TensorType(n_x.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], TupleType([dx_ty, TensorType([], dtype=m_indices.dtype)]))
    expected_type = FuncType([x_ty, indices_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("max_length", [3, 4, 5, 6])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
@pytest.mark.parametrize("other_feature_dims", [[1, 2], [3, 4], [5, 6]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sequence_mask(max_length, batch_size, other_feature_dims,
                       axis, dtype):
    import tvm.topi.testing as npx

    class SequenceMask(mnm.Model):
        def build(self, mask_value, axis=0):
            self._axis = axis
            self._mask_value = mask_value

        @mnm.model.trace
        def forward(self, x, sequence_length):
            return mnm.sequence_mask(x, sequence_length,
                                     mask_value=self._mask_value, axis=self._axis)

    x_shape = [max_length, batch_size] if axis == 0 else [batch_size, max_length]
    x_shape += other_feature_dims
    model = SequenceMask(-10, axis)
    # forward
    m_x, n_x = randn(x_shape, dtype=dtype)
    m_length, n_length = randint([batch_size], low=0, high=max_length, dtype=dtype)
    m_func = model.get_relay_func(m_x, m_length)
    m_func = run_infer_type(m_func)
    n_y = npx.sequence_mask(n_x, n_length, axis=axis, mask_value=-10)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    length_ty = TensorType(n_length.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, length_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [[10, 10, 10], [6, 8, 9, 10]])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reverse(shape, axis, dtype):

    class Reverse(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.reverse(x, self._axis)

    m_x, n_x = randn(shape, dtype=dtype)
    model = Reverse(axis=axis)
    # forward
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = np.flip(n_x, axis=axis)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("inputs", [
    {"shape": (3, 3, 3), "seq_length": [1, 2, 3]},
    {"shape": (5, 5, 5), "seq_length": [1, 2, 3, 4, 5]},
    {"shape": (5, 5, 5), "seq_length": [2, 2, 3, 3, 4]}
])
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
    # forward
    m_func = model.get_relay_func(m_x, m_seq_length)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=dtype)
    seq_length_ty = TensorType(m_seq_length.shape, dtype="int64")
    y_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty, seq_length_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], TupleType([dx_ty, TensorType([], dtype="int64")]))
    expected_type = FuncType([x_ty, seq_length_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [
    [[1, 4, 1], [1, 4, 1]],
    [[1, 4, 1], [1, 2, 4, 1]],
    [[4, 1, 1], [3, 4, 2, 2]]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_to(shape, dtype):

    class BroadcastTo(mnm.Model):
        def build(self, shape=None):
            self._shape = shape

        @mnm.model.trace
        def forward(self, x):
            return mnm.broadcast_to(x, self._shape)

    model = BroadcastTo(shape[1])
    m_x, _ = randn(shape[0], dtype=dtype)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape[0], dtype=dtype)
    y_ty = TensorType(shape[1], dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [
    [[1, 4, 1], [1, 4, 1]],
    [[1, 4, 1], [1, 2, 4, 1]],
    [[4, 1, 1], [3, 4, 2, 2]]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_to_like(shape, dtype):

    class BroadcastToLike(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, broadcast_type): # pylint: disable=no-self-use
            return mnm.broadcast_to_like(x, broadcast_type)

    model = BroadcastToLike()
    m_x, _ = randn(shape[0], dtype=dtype)
    broadcast_type, _ = randn(shape[1], dtype=dtype)
    m_func = model.get_relay_func(m_x, broadcast_type)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape[0], dtype=dtype)
    broadcast_ty = TensorType(shape[1], dtype=dtype)
    y_ty = TensorType(shape[1], dtype=dtype)
    expected_type = FuncType([x_ty, broadcast_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [
    (1, 4, 1),
    (3, 4, 2, 2),
    (4, 1, 1),
    (1, 2, 4, 1)
])
@pytest.mark.parametrize("repeats", [0, 1, 2])
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
def test_repeat(shape, repeats, axis, dtype):

    class Repeat(mnm.Model):
        def build(self, repeats, axis=0):
            self._repeats = repeats
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            return mnm.repeat(x, repeats=self._repeats, axis=self._axis)

    model = Repeat(repeats, axis)
    # forward
    m_x, n_x = randn(shape, dtype=dtype)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = np.repeat(n_x, repeats, axis)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("params", [
    {"shapes": [[1, 4, 1]], "axis": 0},
    {"shapes": [[1, 4, 1], [1, 4, 1]], "axis": 0},
    {"shapes": [[2, 2, 2], [2, 2, 2], [2, 2, 2]], "axis": -1},
    {"shapes": [[2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1]], "axis": 1},
])
def test_stack(params, dtype):

    class Stack1(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a):
            return mnm.stack([a], axis=self._axis)

    class Stack2(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b):
            return mnm.stack([a, b], axis=self._axis)

    class Stack3(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.stack([a, b, c], axis=self._axis)

    class Stack4(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c, d):
            return mnm.stack([a, b, c, d], axis=self._axis)

    stack = [None, Stack1, Stack2, Stack3, Stack4]
    shapes, axis = params["shapes"], params["axis"]
    m_i, n_i, i_ty = [], [], []
    for shape in shapes:
        m_x, n_x = randn(shape, dtype=dtype)
        x_ty = TensorType(shape, dtype=dtype)
        m_i.append(m_x)
        n_i.append(n_x)
        i_ty.append(x_ty)
    model = stack[len(m_i)](axis=axis) # pylint: disable=not-callable
    # forward
    m_func = model.get_relay_func(*m_i)
    m_func = run_infer_type(m_func)
    n_y = np.stack(n_i, axis=axis)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType(i_ty, y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [[10, 20, 30], [6, 8, 10, 3]])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("indices_or_sections", [(2, 4), (1, 4), 2, (2,)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_split(shape, axis, indices_or_sections, dtype):

    class Split(mnm.Model):
        def build(self, indices_or_sections, axis):
            self._indices_or_sections = indices_or_sections
            self._axis = axis

        @mnm.model.trace
        def forward(self, x):
            ret = mnm.split(x, self._indices_or_sections, self._axis)
            return ret

    m_x, n_x = randn(shape, dtype=dtype)
    n_y = np.split(n_x, indices_or_sections=indices_or_sections, axis=axis)
    # forward
    model = Split(indices_or_sections, axis)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = []
    for n_i in n_y:
        i_ty = TensorType(n_i.shape, dtype=dtype)
        y_ty.append(i_ty)
    expected_type = FuncType([x_ty], TupleType(y_ty))
    check_type(m_func, expected_type)


# pylint: disable=too-many-locals, attribute-defined-outside-init
@pytest.mark.parametrize("shape", [
    [(2, 2), (1, 0)],
    [(2, 2), None],
    [(2, 2, 2), (1, 2, 0)],
    [(2, 2, 2), (2, 1, 0)],
    [(2, 2, 2), None],
    [(4, 4, 4, 4), (3, 2, 1, 0)],
    [(4, 4, 4, 4), (1, 2, 3, 0)]
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
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
    m_x, n_x = randn(shape[0], dtype=dtype)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = np.transpose(n_x, shape[1])
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    # forward
    check_type(m_func, expected_type)
    # backward
    y_shape = n_y.shape
    _, n_dy = randn(y_shape, dtype=dtype)
    if axes is not None:
        axes_inverse = list(axes).copy()
        for idx, i in enumerate(list(axes)):
            axes_inverse[i] = idx
        n_x_grad = np.transpose(n_dy, axes=tuple(axes_inverse))
    else:
        n_x_grad = np.transpose(n_dy)
    dy_ty = TensorType(n_dy.shape, dtype=dtype)
    dx_ty = TensorType(n_x_grad.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [(1, 2), (3, 4, 2), (1, 5, 3), (2, 0)])
@pytest.mark.parametrize("itype", ["float16", "float32", "float64", "int32", "int64", "bool"])
@pytest.mark.parametrize("otype", ["float16", "float32", "float64", "int32", "int64", "bool"])
def test_cast(shape, itype, otype):

    class Cast(mnm.Model):
        def build(self, otype=None):
            self._otype = otype

        @mnm.model.trace
        def forward(self, data):
            return mnm.cast(data, self._otype)

    m_x, n_x = randn(shape, dtype=itype)
    model = Cast(otype)
    # forward
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = n_x.astype(otype)
    x_ty = TensorType(n_x.shape, dtype=itype)
    y_ty = TensorType(n_y.shape, dtype=otype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    dy_ty = TensorType(n_y.shape, dtype=otype)
    dx_ty = TensorType(n_x.shape, dtype=itype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [(1, 2), (3, 4, 2), (1, 5, 3), (2, 0)])
@pytest.mark.parametrize("itype", ["float16", "float32", "float64", "int32", "int64", "bool"])
@pytest.mark.parametrize("otype", ["float16", "float32", "float64", "int32", "int64", "bool"])
def test_cast_like(shape, itype, otype):

    class CastLike(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, data, dtype_like): # pylint: disable=no-self-use
            return mnm.cast_like(data, dtype_like)

    m_x, _ = randn(shape, dtype=itype)
    m_dtype_like, _ = randn(shape, dtype=otype)
    model = CastLike()
    m_func = model.get_relay_func(m_x, m_dtype_like)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=itype)
    dtype_like_ty = TensorType(shape, dtype=otype)
    y_ty = TensorType(shape, dtype=otype)
    expected_type = FuncType([x_ty, dtype_like_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("params", [
    {"shapes": [[1, 4, 1]], "axis": 0},
    {"shapes": [[1, 4, 1], [2, 4, 1]], "axis": 0},
    {"shapes": [[2, 2, 2], [2, 3, 2], [2, 4, 2]], "axis": -2},
    {"shapes": [[2, 1, 1], [2, 2, 1], [2, 3, 1], [2, 4, 1]], "axis": 1},
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_concatenate(params, dtype):

    class Concatenate1(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a):
            return mnm.concatenate([a], axis=self._axis)

    class Concatenate2(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b):
            return mnm.concatenate([a, b], axis=self._axis)

    class Concatenate3(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c):
            return mnm.concatenate([a, b, c], axis=self._axis)

    class Concatenate4(mnm.Model):
        def build(self, axis):
            self._axis = axis

        @mnm.model.trace
        def forward(self, a, b, c, d):
            return mnm.concatenate([a, b, c, d], axis=self._axis)

    concat = [None, Concatenate1, Concatenate2, Concatenate3, Concatenate4]
    # forward
    shapes, axis = params["shapes"], params["axis"]
    m_i, t_i, i_ty = [], [], []
    for shape in shapes:
        m_x, t_x = randn_torch(shape, dtype=dtype)
        x_ty = TensorType(shape, dtype=dtype)
        m_i.append(m_x)
        t_i.append(t_x)
        i_ty.append(x_ty)
    model = concat[len(m_i)](axis=axis) # pylint: disable=not-callable
    m_func = model.get_relay_func(*m_i)
    m_func = run_infer_type(m_func)
    t_y = torch.cat(t_i, dim=axis) # pylint: disable=no-member
    y_ty = TensorType(t_y.shape, dtype=dtype)
    expected_type = FuncType(i_ty, y_ty)
    check_type(m_func, expected_type)


# pylint: disable=no-self-use
@pytest.mark.parametrize("shape", [(1, 3), (1, 2), (4, 3, 2, 1),
                                   (2, 4, 1, 3), (1, 2, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("a_min", [0.1, 0.3, 0.4])
@pytest.mark.parametrize("a_max", [0.6, 0.7, 0.8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_clip(shape, a_min, a_max, dtype):

    class Clip(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.clip(x, a_min, a_max)

    m_x, n_x = randn(shape, dtype=dtype)
    model = Clip()
    # forward
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = np.clip(n_x, a_min, a_max)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)
    # check backward
    m_func = AutoDiff(m_func)
    m_func = run_infer_type(m_func)
    bwd_ty = FuncType([y_ty], x_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func, expected_type)


@pytest.mark.parametrize("params", [
    {"orig_shape": (8, 8, 8, 8), "to_shape": (2, 2048),
     "infer_shape": (2, 2048), "reverse_infer_shape": (2, 2048)},
    {"orig_shape": (8, 1000), "to_shape": (2, 2, 2, 1000),
     "infer_shape": (2, 2, 2, 1000), "reverse_infer_shape": (2, 2, 2, 1000)},
    {"orig_shape": (3, 3, 3, 3), "to_shape": (81, 1),
     "infer_shape": (81, 1), "reverse_infer_shape": (81, 1)},
    {"orig_shape": (3, 3, 3, 3), "to_shape": (0, -1),
     "infer_shape": (3, 27), "reverse_infer_shape": (3, 27)},
    {"orig_shape": (2, 3, 4, 5), "to_shape": (0, 0, -1),
     "infer_shape": (2, 3, 20), "reverse_infer_shape": (3, 4, 10)},
])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reshape(params, reverse, dtype):

    class Reshape(mnm.Model):
        def build(self, shape, reverse=False):
            self._shape = shape
            self._reverse = reverse

        @mnm.model.trace
        def forward(self, x):
            return mnm.reshape(x, shape=self._shape, reverse=self._reverse)

    orig_shape, to_shape, infer_shape, reverse_infer_shape = \
    params["orig_shape"], params["to_shape"], params["infer_shape"], params["reverse_infer_shape"]
    model = Reshape(shape=to_shape, reverse=reverse)
    # forward
    m_x, _ = randn(orig_shape, dtype=dtype)
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    x_ty = TensorType(orig_shape, dtype=dtype)
    if reverse:
        y_ty = TensorType(reverse_infer_shape, dtype=dtype)
    else:
        y_ty = TensorType(infer_shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)
    # check backward
    # TODO(@XIAO-XIA): uncomment after impl the type funcs of shape
    # m_func = AutoDiff(m_func)
    # m_func = run_infer_type(m_func)
    # bwd_ty = FuncType([y_ty], x_ty)
    # expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    # check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [
    [10, 3, 2, 5],
    [1, 4, 5, 2],
    [9, 12, 18, 2, 1]])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
@pytest.mark.parametrize("num_newaxis", [0, 1, 2, 5])
def test_expand_dims(shape, dtype, axis, num_newaxis):

    class ExpandDims(mnm.Model):
        def build(self, axis, num_newaxis):
            self.axis = axis
            self.num_newaxis = num_newaxis

        @mnm.model.trace
        def forward(self, x):
            return mnm.expand_dims(x, axis=self.axis, num_newaxis=self.num_newaxis)

    m_x, n_x = randn(shape, dtype=dtype)
    model = ExpandDims(axis, num_newaxis)
    # forward
    m_func = model.get_relay_func(m_x)
    m_func = run_infer_type(m_func)
    n_y = n_x
    if num_newaxis == 0:
        pass
    elif num_newaxis == 1:
        n_y = np.expand_dims(n_y, axis=axis)
    else:
        for _ in range(num_newaxis):
            n_y = np.expand_dims(n_y, axis=axis)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)
    # backward
    # TODO(@XIAO-XIA): uncomment after impl the type funcs of shape and reshape
    # bwd_ty = FuncType([y_ty], x_ty)
    # expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    # m_func = AutoDiff(m_func)
    # m_func = run_infer_type(m_func)
    # check_type(m_func, expected_type)


if __name__ == "__main__":
    pytest.main([__file__])