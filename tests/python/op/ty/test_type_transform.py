# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, too-many-lines
import numpy as np
import pytest
import torch
import raf
import tvm.topi.testing as npx
from raf._ffi.pass_ import AutoDiff, InferType
from raf.testing import check_type, run_infer_type, randn, randn_torch, randint
from tvm.relay import TensorType, FuncType, TupleType, Any


# pylint: disable=too-many-locals, import-outside-toplevel, attribute-defined-outside-init
@pytest.mark.parametrize(
    "shape",
    [
        [(5, 4, 3), (1, 2)],
        [(6, 5), (2, 2)],
    ],
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_take(shape, axis, dtype):
    from functools import reduce
    import operator

    class Take(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, x, indices):
            return raf.take(x, indices=indices, axis=self._axis)

    size = reduce(operator.mul, shape[0], 1) if axis is None else shape[0][axis]
    m_x, n_x = randn(shape[0], dtype=dtype)
    m_x.requires_grad = True
    m_indices, n_indices = randint(shape[1], low=0, high=size)
    model = Take(axis)
    # forward
    record = model._internal(m_x, m_indices)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    n_y = np.take(n_x, n_indices, axis=axis, mode="clip")
    x_ty = TensorType(n_x.shape, dtype=dtype)
    indices_ty = TensorType(n_indices.shape, dtype=m_indices.dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, indices_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    dy_ty = TensorType(n_y.shape, dtype=dtype)
    dx_ty = TensorType(n_x.shape, dtype=dtype)
    bwd_ty = FuncType([dy_ty], TupleType([dx_ty, TensorType([], dtype=m_indices.dtype)]))
    expected_type = FuncType([x_ty, indices_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("max_length", [3, 4, 5, 6])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
@pytest.mark.parametrize("other_feature_dims", [[1, 2], [5, 6]])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_sequence_mask(max_length, batch_size, other_feature_dims, axis, dtype):
    class SequenceMask(raf.Model):
        def build(self, mask_value, axis=0):
            self._axis = axis
            self._mask_value = mask_value

        @raf.model.trace
        def forward(self, x, sequence_length):
            return raf.sequence_mask(
                x, sequence_length, mask_value=self._mask_value, axis=self._axis
            )

    x_shape = [max_length, batch_size] if axis == 0 else [batch_size, max_length]
    x_shape += other_feature_dims
    model = SequenceMask(-10, axis)
    # forward
    m_x, n_x = randn(x_shape, dtype=dtype)
    m_length, n_length = randint([batch_size], low=0, high=max_length, dtype=dtype)
    m_mod = model._internal(m_x, m_length).mod
    m_mod = InferType()(m_mod)
    n_y = npx.sequence_mask(n_x, n_length, axis=axis, mask_value=-10)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    length_ty = TensorType(n_length.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, length_ty], y_ty)
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("shape", [[10, 10, 10], [6, 8, 9, 10]])
@pytest.mark.parametrize("axis", [0, 2])
def test_reverse(shape, axis, dtype):
    class Reverse(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, x):
            return raf.reverse(x, self._axis)

    m_x, n_x = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    model = Reverse(axis=axis)
    # forward
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    n_y = np.flip(n_x, axis=axis)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "inputs",
    [
        {"shape": (3, 3, 3), "seq_length": [1, 2, 3]},
        {"shape": (5, 5, 5), "seq_length": [2, 2, 3, 3, 4]},
    ],
)
@pytest.mark.parametrize("axes", [[0, 1]])
def test_reverse_sequence(inputs, axes, dtype):
    class ReverseSequence(raf.Model):
        def build(self, seq_axis, batch_axis):
            self._seq_axis = seq_axis
            self._batch_axis = batch_axis

        @raf.model.trace
        def forward(self, x, seq_length):
            return raf.reverse_sequence(x, seq_length, self._seq_axis, self._batch_axis)

    shape = inputs["shape"]
    m_seq_length = raf.array(inputs["seq_length"], dtype="int64")
    seq_axis = axes[0]
    batch_axis = axes[1]
    m_x, _ = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    model = ReverseSequence(seq_axis, batch_axis)
    # forward
    record = model._internal(m_x, m_seq_length)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape, dtype=dtype)
    seq_length_ty = TensorType(m_seq_length.shape, dtype="int64")
    y_ty = TensorType(shape, dtype=dtype)
    expected_type = FuncType([x_ty, seq_length_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], TupleType([dx_ty, TensorType([], dtype="int64")]))
    expected_type = FuncType([x_ty, seq_length_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("shape", [[[1, 4, 1], [1, 4, 1]], [[4, 1, 1], [3, 4, 2, 2]]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_broadcast_to(shape, dtype):
    class BroadcastTo(raf.Model):
        def build(self, shape=None):
            self._shape = shape

        @raf.model.trace
        def forward(self, x):
            return raf.broadcast_to(x, self._shape)

    model = BroadcastTo(shape[1])
    m_x = randn_torch(shape[0], requires_grad=True, dtype=dtype)[0]
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape[0], dtype=dtype)
    y_ty = TensorType(shape[1], dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)

    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = run_infer_type(m_mod)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("shape", [[[1, 4, 1], [1, 4, 1]], [[4, 1, 1], [3, 4, 2, 2]]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_broadcast_to_like(shape, dtype):
    class BroadcastToLike(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, broadcast_type):  # pylint: disable=no-self-use
            return raf.broadcast_to_like(x, broadcast_type)

    model = BroadcastToLike()
    m_x, _ = randn(shape[0], dtype=dtype)
    broadcast_type, _ = randn(shape[1], dtype=dtype)
    m_func = model._internal(m_x, broadcast_type).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape[0], dtype=dtype)
    broadcast_ty = TensorType(shape[1], dtype=dtype)
    y_ty = TensorType(shape[1], dtype=dtype)
    expected_type = FuncType([x_ty, broadcast_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [[[1, 4, 1], [1, 4, 1]], [[4, 1, 1], [3, 4, 2, 2]]])
def test_collapse_sum_like(shape):
    class CollapseSumLikeModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, like_type):  # pylint: disable=no-self-use
            return raf.collapse_sum_like(x, like_type)

    model = CollapseSumLikeModel()
    m_x, _ = randn(shape[0])
    like_type, _ = randn(shape[1])
    m_func = model._internal(m_x, like_type).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape[0])
    like_ty = TensorType(shape[1])
    y_ty = TensorType(shape[1])
    expected_type = FuncType([x_ty, like_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [[[1, 4, 1], [2, 2]], [[4, 1, 1], [1, 2, 2]]])
def test_reshape_like(shape):
    class ReshapeLikeModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, like_type):  # pylint: disable=no-self-use
            return raf.reshape_like(x, like_type)

    model = ReshapeLikeModel()
    m_x, _ = randn(shape[0])
    like_type, _ = randn(shape[1])
    m_func = model._internal(m_x, like_type).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape[0])
    like_ty = TensorType(shape[1])
    y_ty = TensorType(shape[1])
    expected_type = FuncType([x_ty, like_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 1),
        (3, 4, 2, 2),
    ],
)
@pytest.mark.parametrize("repeats", [0, 1, 2])
@pytest.mark.parametrize("axis", [-1, 0, 2])
def test_repeat(shape, repeats, axis, dtype):
    class Repeat(raf.Model):
        def build(self, repeats, axis=0):
            self._repeats = repeats
            self._axis = axis

        @raf.model.trace
        def forward(self, x):
            return raf.repeat(x, repeats=self._repeats, axis=self._axis)

    model = Repeat(repeats, axis)
    # forward
    m_x, n_x = randn(shape, dtype=dtype, requires_grad=True)
    record = model._internal(m_x)
    m_mod = record.mod
    m_func = m_mod["main"]
    m_func = run_infer_type(m_func)
    n_y = np.repeat(n_x, repeats, axis)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_func, expected_type)

    # backward
    m_func = InferType()(m_mod)
    m_func = AutoDiff(record.requires_grads)(m_func)
    m_func = run_infer_type(m_func)
    dy_ty, dx_ty = y_ty, x_ty
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_func["main"], expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "params",
    [
        {"shapes": [[1, 4, 1]], "axis": 0},
        {"shapes": [[1, 4, 1], [1, 4, 1]], "axis": 0},
        {"shapes": [[2, 2, 2], [2, 2, 2], [2, 2, 2]], "axis": -1},
    ],
)
def test_stack(params, dtype):
    class Stack1(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a):
            return raf.stack([a], axis=self._axis)

    class Stack2(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b):
            return raf.stack([a, b], axis=self._axis)

    class Stack3(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b, c):
            return raf.stack([a, b, c], axis=self._axis)

    class Stack4(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b, c, d):
            return raf.stack([a, b, c, d], axis=self._axis)

    stack = [None, Stack1, Stack2, Stack3, Stack4]
    shapes, axis = params["shapes"], params["axis"]
    m_i, n_i, i_ty = [], [], []
    for shape in shapes:
        m_x, n_x = randn(shape, dtype=dtype)
        x_ty = TensorType(shape, dtype=dtype)
        m_i.append(m_x)
        n_i.append(n_x)
        i_ty.append(x_ty)
    model = stack[len(m_i)](axis=axis)  # pylint: disable=not-callable
    # forward
    m_func = model._internal(*m_i).mod["main"]
    m_func = run_infer_type(m_func)
    n_y = np.stack(n_i, axis=axis)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType(i_ty, y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize("shape", [[10, 20, 30], [6, 8, 10, 3]])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("indices_or_sections", [(2, 4), (1, 4), 2, (2,)])
@pytest.mark.parametrize("dtype", ["float32"])
def test_split(shape, axis, indices_or_sections, dtype):
    class Split(raf.Model):
        def build(self, indices_or_sections, axis):
            self._indices_or_sections = indices_or_sections
            self._axis = axis

        @raf.model.trace
        def forward(self, x):
            ret = raf.split(x, self._indices_or_sections, self._axis)
            return ret

    m_x, n_x = randn(shape, dtype=dtype)
    n_y = np.split(n_x, indices_or_sections=indices_or_sections, axis=axis)
    # forward
    model = Split(indices_or_sections, axis)
    m_func = model._internal(m_x).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = []
    for n_i in n_y:
        i_ty = TensorType(n_i.shape, dtype=dtype)
        y_ty.append(i_ty)
    expected_type = FuncType([x_ty], TupleType(y_ty))
    check_type(m_func, expected_type)


# pylint: disable=too-many-locals, attribute-defined-outside-init
@pytest.mark.parametrize(
    "shape",
    [
        [(2, 2), (1, 0)],
        [(2, 2), None],
        [(2, 2, 2), (1, 2, 0)],
        [(2, 2, 2), None],
        [(4, 4, 4, 4), (1, 2, 3, 0)],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_transpose(shape, dtype):
    class Transpose(raf.Model):
        def build(self, axes=None):
            self._axes = axes

        @raf.model.trace
        def forward(self, x):
            ret = raf.transpose(x, self._axes)
            return ret

    axes = shape[1]
    model = Transpose(axes)
    m_x, n_x = randn(shape[0], dtype=dtype)
    m_x.requires_grad = True
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    n_y = np.transpose(n_x, shape[1])
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    # forward
    check_type(m_mod["main"], expected_type)
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
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2),
        (6, 2, 5),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        (0, 1),
        (1, 3),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_swap_axis(shape, dtype, axis):
    class SwapAxis(raf.Model):
        def build(self, axes=None):
            self._axes1 = axes[0]
            self._axes2 = axes[1]

        @raf.model.trace
        def forward(self, x):
            ret = raf.swap_axis(x, self._axes1, self._axes2)
            return ret

    if max(axis) < len(shape):
        axes = axis
        model = SwapAxis(axes)
        m_x, n_x = randn(shape, dtype=dtype)
        m_x.requires_grad = True
        record = model._internal(m_x)
        m_mod = record.mod
        m_func = InferType()(m_mod)
        n_y = np.swapaxes(n_x, axes[0], axes[1])
        x_ty = TensorType(n_x.shape, dtype=dtype)
        y_ty = TensorType(n_y.shape, dtype=dtype)
        expected_type = FuncType([x_ty], y_ty)
        # forward
        check_type(m_func["main"], expected_type)
        # backward
        y_shape = n_y.shape
        _, n_dy = randn(y_shape, dtype=dtype)
        n_x_grad = np.swapaxes(n_dy, axes[0], axes[1])
        dy_ty = TensorType(n_dy.shape, dtype=dtype)
        dx_ty = TensorType(n_x_grad.shape, dtype=dtype)
        bwd_ty = FuncType([dy_ty], dx_ty)
        expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
        m_func = AutoDiff(record.requires_grads)(m_func)
        m_func = run_infer_type(m_func)
        check_type(m_func["main"], expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "dtype_int",
    [
        ["int64", torch.int64],  # only int64 is used since torch scatter only accrpt int64
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "shape",
    [
        [(3, 5), (2, 4), (2, 4)],
        [(3, 5), (3, 5), (3, 5)],
    ],
)  # pylint: disable-msg=too-many-locals, not-callable
def test_scatter(shape, axis, dtype_int, dtype):
    class Scatter(raf.Model):
        def build(self):
            self.axis = axis

        @raf.model.trace
        def forward(self, x, idx, src):
            ret = raf.scatter(x, self.axis, idx, src)
            return ret

    m_x, t_x = randn_torch(shape[0], requires_grad=True, dtype=dtype)
    # dtype would be tested on ty, here using int64 since torch.scatter as require so
    index_shape = list(shape[1])
    del index_shape[axis]
    dim_range = list(range(len(shape[1])))
    del dim_range[axis]
    random_index = np.random.choice(shape[0][axis], shape[1][axis], replace=False)
    chose_index = np.broadcast_to(random_index, tuple(index_shape + [shape[1][axis]]))
    chose_index = np.swapaxes(chose_index, axis, len(shape[1]) - 1)
    t_idx = torch.tensor(chose_index, dtype=dtype_int[1])
    m_idx = raf.array(chose_index, dtype=dtype_int[0])
    m_src, t_src = randn_torch(shape[2], dtype=dtype)
    m_x.requires_grad = True
    t_x.requires_grad = True

    model = Scatter()
    record = model._internal(m_x, m_idx, m_src)
    m_mod = record.mod
    m_func = InferType()(m_mod)

    t_y = torch.scatter(t_x, axis, t_idx, t_src)
    x_ty = TensorType(t_x.shape, dtype=dtype)
    src_ty = TensorType(t_src.shape, dtype=dtype)
    idx_ty = TensorType(t_idx.shape, dtype=dtype_int[0])
    y_ty = TensorType(t_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty, idx_ty, src_ty], y_ty)

    # forward
    check_type(m_func["main"], expected_type)


@pytest.mark.parametrize("shape", [(1, 2), (3, 4, 2), (2, 0)])
@pytest.mark.parametrize("itype", ["float16", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("otype", ["float16", "float32", "int32", "int64", "bool"])
def test_cast(shape, itype, otype):
    class Cast(raf.Model):
        def build(self, otype=None):
            self._otype = otype

        @raf.model.trace
        def forward(self, data):
            return raf.cast(data, self._otype)

    m_x, n_x = randn(shape, dtype=itype)
    m_x.requires_grad = True
    model = Cast(otype)
    # forward
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    n_y = n_x.astype(otype)
    x_ty = TensorType(n_x.shape, dtype=itype)
    y_ty = TensorType(n_y.shape, dtype=otype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    dy_ty = TensorType(n_y.shape, dtype=otype)
    dx_ty = TensorType(n_x.shape, dtype=itype)
    bwd_ty = FuncType([dy_ty], dx_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("shape", [(1, 2), (3, 4, 2), (2, 0)])
@pytest.mark.parametrize("itype", ["float16", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("otype", ["float16", "float32", "int32", "int64", "bool"])
def test_cast_like(shape, itype, otype):
    class CastLike(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, data, dtype_like):  # pylint: disable=no-self-use
            return raf.cast_like(data, dtype_like)

    m_x, _ = randn(shape, dtype=itype)
    m_dtype_like, _ = randn(shape, dtype=otype)
    model = CastLike()
    m_func = model._internal(m_x, m_dtype_like).mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(shape, dtype=itype)
    dtype_like_ty = TensorType(shape, dtype=otype)
    y_ty = TensorType(shape, dtype=otype)
    expected_type = FuncType([x_ty, dtype_like_ty], y_ty)
    check_type(m_func, expected_type)


@pytest.mark.parametrize(
    "params",
    [
        {"shapes": [[1, 4, 1]], "axis": 0},
        {"shapes": [[1, 4, 1], [2, 4, 1]], "axis": 0},
        {"shapes": [[2, 2, 2], [2, 3, 2], [2, 4, 2]], "axis": -2},
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_concatenate(params, dtype):
    class Concatenate1(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a):
            return raf.concatenate([a], axis=self._axis)

    class Concatenate2(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b):
            return raf.concatenate([a, b], axis=self._axis)

    class Concatenate3(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b, c):
            return raf.concatenate([a, b, c], axis=self._axis)

    class Concatenate4(raf.Model):
        def build(self, axis):
            self._axis = axis

        @raf.model.trace
        def forward(self, a, b, c, d):
            return raf.concatenate([a, b, c, d], axis=self._axis)

    concat = [None, Concatenate1, Concatenate2, Concatenate3, Concatenate4]
    # forward
    shapes, axis = params["shapes"], params["axis"]
    m_i, t_i, i_ty = [], [], []
    for shape in shapes:
        m_x, t_x = randn_torch(shape, dtype=dtype)
        m_x.requires_grad = True
        x_ty = TensorType(shape, dtype=dtype)
        m_i.append(m_x)
        t_i.append(t_x)
        i_ty.append(x_ty)
    model = concat[len(m_i)](axis=axis)  # pylint: disable=not-callable
    record = model._internal(*m_i)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    t_y = torch.cat(t_i, dim=axis)  # pylint: disable=no-member
    y_ty = TensorType(t_y.shape, dtype=dtype)
    expected_type = FuncType(i_ty, y_ty)
    check_type(m_mod["main"], expected_type)
    # backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([y_ty], TupleType(i_ty) if len(i_ty) > 1 else i_ty[0])
    expected_type = FuncType(i_ty, TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


# pylint: disable=no-self-use
@pytest.mark.parametrize(
    "shapes",
    [
        [1, 5, 2],
        [6, 3],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
def test_mesh_grid(shapes, dtype):
    class MeshGrid2(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, a, b):
            return raf.mesh_grid([a, b])

    class MeshGrid3(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, a, b, c):
            return raf.mesh_grid([a, b, c])

    class MeshGrid4(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, a, b, c, d):
            return raf.mesh_grid([a, b, c, d])

    # one input is trivial case
    meshgrid = [None, None, MeshGrid2, MeshGrid3, MeshGrid4]

    # forward
    m_i, t_i, i_ty = [], [], []
    for shape in shapes:
        m_x, t_x = randn_torch([shape], requires_grad=True, dtype=dtype)
        m_x.requires_grad = True
        x_ty = TensorType([shape], dtype=dtype)
        m_i.append(m_x)
        t_i.append(t_x)
        i_ty.append(x_ty)

    model = meshgrid[len(m_i)]()  # pylint: disable=not-callable
    record = model._internal(*m_i)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    t_y = torch.meshgrid(t_i)  # pylint: disable=no-member

    y_ty = []
    for t_i in t_y:
        y_ty.append(TensorType(t_i.shape, dtype=dtype))

    expected_type = FuncType(i_ty, TupleType(y_ty))
    check_type(m_mod["main"], expected_type)


# pylint: disable=no-self-use
@pytest.mark.parametrize("shape", [(1, 3), (2, 4, 1, 3)])
@pytest.mark.parametrize("a_min", [0.1, 0.3, 0.4])
@pytest.mark.parametrize("a_max", [0.6, 0.7, 0.8])
@pytest.mark.parametrize("dtype", ["float32"])
def test_clip(shape, a_min, a_max, dtype):
    class Clip(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.clip(x, a_min, a_max)

    m_x, n_x = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    model = Clip()
    # forward
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    n_y = np.clip(n_x, a_min, a_max)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([y_ty], x_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize(
    "params",
    [
        {
            "orig_shape": (8, 8, 8, 8),
            "to_shape": (2, 2048),
            "infer_shape": (2, 2048),
            "reverse_infer_shape": (2, 2048),
        },
        {
            "orig_shape": (8, 1000),
            "to_shape": (2, 2, 2, 1000),
            "infer_shape": (2, 2, 2, 1000),
            "reverse_infer_shape": (2, 2, 2, 1000),
        },
        {
            "orig_shape": (3, 3, 3, 3),
            "to_shape": (81, 1),
            "infer_shape": (81, 1),
            "reverse_infer_shape": (81, 1),
        },
        {
            "orig_shape": (3, 3, 3, 3),
            "to_shape": (0, -1),
            "infer_shape": (3, 27),
            "reverse_infer_shape": (3, 27),
        },
        {
            "orig_shape": (2, 3, 4, 5),
            "to_shape": (0, 0, -1),
            "infer_shape": (2, 3, 20),
            "reverse_infer_shape": (3, 4, 10),
        },
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", ["float32"])
def test_reshape(params, reverse, dtype):
    class Reshape(raf.Model):
        def build(self, shape, reverse=False):
            self._shape = shape
            self._reverse = reverse

        @raf.model.trace
        def forward(self, x):
            return raf.reshape(x, shape=self._shape, reverse=self._reverse)

    orig_shape, to_shape, infer_shape, reverse_infer_shape = (
        params["orig_shape"],
        params["to_shape"],
        params["infer_shape"],
        params["reverse_infer_shape"],
    )
    model = Reshape(shape=to_shape, reverse=reverse)
    # forward
    m_x, _ = randn(orig_shape, dtype=dtype)
    m_x.requires_grad = True
    record = model._internal(m_x)  # pylint: disable=unused-variable
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(orig_shape, dtype=dtype)
    if reverse:
        y_ty = TensorType(reverse_infer_shape, dtype=dtype)
    else:
        y_ty = TensorType(infer_shape, dtype=dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)
    # check backward
    # TODO(@XIAO-XIA): uncomment after impl the type funcs of shape
    # m_mod = AutoDiff(record.requires_grads)(m_mod)
    # m_mod = InferType()(m_mod)
    # bwd_ty = FuncType([y_ty], x_ty)
    # expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    # check_type(m_mod['main], expected_type)


@pytest.mark.parametrize("shape", [[1, 4, 5, 2], [9, 12, 18, 2, 1]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
@pytest.mark.parametrize("num_newaxis", [0, 1, 2, 5])
def test_expand_dims(shape, dtype, axis, num_newaxis):
    class ExpandDims(raf.Model):
        def build(self, axis, num_newaxis):
            self.axis = axis
            self.num_newaxis = num_newaxis

        @raf.model.trace
        def forward(self, x):
            return raf.expand_dims(x, axis=self.axis, num_newaxis=self.num_newaxis)

    m_x, n_x = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    model = ExpandDims(axis, num_newaxis)
    # forward
    record = model._internal(m_x)  # pylint: disable=unused-variable
    m_mod = record.mod
    m_mod = InferType()(m_mod)
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
    check_type(m_mod["main"], expected_type)
    # backward
    # TODO(@XIAO-XIA): uncomment after impl the type funcs of shape and reshape
    # bwd_ty = FuncType([y_ty], x_ty)
    # expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    # m_mod = AutoDiff(record.requires_grads)(m_mod)
    # m_mod = InferType()(m_mod)
    # check_type(m_mod['main], expected_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("i_dtype", ["int64"])
@pytest.mark.parametrize("dshape", [[10, 11, 12], [10, 11, 12, 13]])
@pytest.mark.parametrize("ishape", [[3], [3, 2], [4, 5, 3]])
def test_gather_nd(dtype, i_dtype, dshape, ishape):
    # pylint: disable=no-self-use
    class GatherNd(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, data, indices):
            return raf.gather_nd(data, indices)

    model = GatherNd()
    m_x, _ = randn(dshape, dtype=dtype)
    m_x.requires_grad = True
    m_i = randint(ishape, high=dshape[0 : ishape[-1]], dtype=i_dtype)[0]
    m_i = raf.transpose(m_i, axes=[len(ishape) - 1] + list(range(len(ishape) - 1)))
    ty_data = TensorType(dshape, dtype=dtype)
    ty_indices = TensorType(m_i.shape, dtype=i_dtype)
    assert ishape[-1] <= len(dshape)
    odim = len(ishape) - 1 + len(dshape) - ishape[-1]
    oshape = []
    for i in range(odim):
        if i + 1 < len(ishape):
            oshape.append(ishape[i])
        else:
            oshape.append(dshape[i + 1 - len(ishape) + ishape[-1]])
    fwd_ty = TensorType(oshape, dtype=dtype)
    # check forward
    record = model._internal(m_x, m_i)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([ty_data, ty_indices], fwd_ty)
    check_type(m_mod["main"], desired_type)
    # check backward
    m_mod = AutoDiff(record.requires_grads)(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([fwd_ty], TupleType([ty_data, TensorType([], dtype=ty_indices.dtype)]))
    desired_type = FuncType([ty_data, ty_indices], TupleType([fwd_ty, bwd_ty]))
    check_type(m_mod["main"], desired_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "params",
    [
        ((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2]),
        ((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2]),
        ((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1]),
    ],
)
def test_strided_slice(dtype, params):
    class StridedSlice(raf.Model):
        def build(self, begin, end, strides):
            self.begin = begin
            self.end = end
            self.strides = strides

        @raf.model.trace
        def forward(self, data):
            return raf.strided_slice(data, self.begin, self.end, self.strides)

    shape, begin, end, strides = params
    model = StridedSlice(begin, end, strides)
    m_x, n_x = randn(shape, dtype=dtype)
    n_y = npx.strided_slice_python(n_x, begin, end, strides)

    m_mod = model._internal(m_x).mod
    m_mod = InferType()(m_mod)
    m_func = m_mod["main"]
    m_func = run_infer_type(m_func)
    x_ty = TensorType(n_x.shape, dtype=dtype)
    y_ty = TensorType(n_y.shape, dtype=dtype)
    desired_type = FuncType([x_ty], y_ty)
    check_type(m_func, desired_type)

    # check backward
    m_mod = AutoDiff([])(m_mod)
    m_mod = InferType()(m_mod)
    bwd_ty = FuncType([y_ty], x_ty)
    expected_type = FuncType([x_ty], TupleType([y_ty, bwd_ty]))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("dtype", ["float32", "int64"])
@pytest.mark.parametrize("data", [[1, 10, 2], [1, 10, 1]])
def test_arange(data, dtype):
    class Arange(raf.Model):
        def build(self, dtype):
            self.dtype = dtype

        @raf.model.trace
        def forward(self, start, stop, step):
            return raf.arange(start, stop, step, self.dtype)

    start, stop, step = data
    m_start = raf.array(start, dtype=dtype)
    m_stop = raf.array(stop, dtype=dtype)
    m_step = raf.array(step, dtype=dtype)
    model = Arange(dtype)
    m_mod = model._internal(m_start, m_stop, m_step).mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType([], dtype=dtype)
    expected_type = FuncType([x_ty, x_ty, x_ty], TensorType([Any()], dtype=dtype))
    check_type(m_mod["main"], expected_type)


@pytest.mark.parametrize("shape", [(), (1,), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "int64", "int32", "bool"])
def test_full(shape, dtype):
    # pylint: disable=invalid-name, attribute-defined-outside-init
    class FullModel(raf.Model):
        def build(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

        @raf.model.trace
        def forward(self):
            return raf.full(fill_value=0, shape=self.shape, dtype=self.dtype)

    model = FullModel(shape, dtype)
    m_func = model._internal().mod["main"]
    m_func = run_infer_type(m_func)
    ty = TensorType(shape, dtype=dtype)
    desired_type = FuncType([], ty)
    check_type(m_func, desired_type)


@pytest.mark.parametrize("shape", [(), (1,), (1, 2, 3, 4)])
@pytest.mark.parametrize("dtype", ["float32", "int64", "int32", "bool"])
def test_full_like(shape, dtype):
    # pylint: disable=invalid-name, attribute-defined-outside-init
    class FullLikeModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, data):
            return raf.full_like(data, fill_value=0)

    m_x, _ = randn(shape, dtype=dtype)
    model = FullLikeModel()
    m_func = model._internal(m_x).mod["main"]
    m_func = run_infer_type(m_func)
    ty = TensorType(shape, dtype=dtype)
    desired_type = FuncType([ty], ty)
    check_type(m_func, desired_type)


@pytest.mark.parametrize(
    "data_shape, index_shapes", [((10, 5), [(1, 4), (3, 1)]), ((10, 5, 4), [(1, 2, 3), (1, 2, 3)])]
)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_adv_index(data_shape, index_shapes, dtype):
    # pylint: disable=invalid-name
    class Index(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, index0, index1):
            return raf.adv_index([x, index0, index1])

    m_x, t_x = randn_torch(data_shape, requires_grad=True, dtype=dtype)
    t_indices = []
    m_indices = []

    for i, index_shape in enumerate(index_shapes):
        limit = data_shape[i]
        index = np.random.uniform(0, limit - 1, size=index_shape).astype("int64")
        t_indices.append(torch.tensor(index))  # pylint: disable=not-callable
        m_indices.append(raf.array(index))
    t_out = t_x[tuple(t_indices)]
    model = Index()
    m_func = model._internal(m_x, m_indices[0], m_indices[1]).mod["main"]
    m_func = run_infer_type(m_func)

    tx = TensorType(m_x.shape, dtype)
    tindex0 = TensorType(m_indices[0].shape, "int64")
    tindex1 = TensorType(m_indices[1].shape, "int64")
    ty = TensorType(t_out.shape, dtype)
    desired_type = FuncType([tx, tindex0, tindex1], ty)
    check_type(m_func, desired_type)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "shape",
    [
        [
            2,
        ],
        [4, 5],
        [6, 8, 9, 10],
    ],
)
def test_argwhere(shape, dtype):
    class Argwhere(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.argwhere(x)

    m_x, _ = randn(shape, dtype=dtype)
    m_x.requires_grad = True
    model = Argwhere()
    # forward
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape, dtype=dtype)
    y_ty = TensorType([Any(), len(shape)], dtype="int32")
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)


if __name__ == "__main__":
    pytest.main([__file__])
