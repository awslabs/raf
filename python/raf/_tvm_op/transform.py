# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, undefined-loop-variable, unused-argument
"""Compute definition and schedules for data transform operators"""
from raf._tvm_op.nn import schedule_generic
from .._lib import register_compute
from .._lib import strategy
from .._lib import tvm as _tvm  # pylint: disable=unused-import
from .._lib import _reg

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member


@register_compute("raf.op.tvm.embedding")
def embedding_compute(attrs, inputs, output_type):
    x, indices = inputs
    return [_topi.take(x, indices, axis=0)]


_reg.register_injective_schedule("raf.op.tvm.embedding")


@register_compute("raf.op.tvm.transpose_dx")
def transpose_dx_compute(attrs, inputs, output_type):
    dy = inputs[0]
    axes = list(_topi.utils.get_const_tuple(attrs.axes))
    axes_inverse = axes.copy()
    for idx, i in enumerate(axes):
        axes_inverse[i] = idx
    out = _topi.transpose(dy, axes=tuple(axes_inverse))
    return [out]


@register_compute("raf.op.tvm.repeat_dx")
def repeat_dx_compute(attrs, inputs, output_type):
    x = inputs[0]
    dy = inputs[1]
    axis = int(attrs.axis)
    shape = x.shape
    split_list = _topi.split(dy, int(shape[axis]), axis)
    result_list = list()
    for item in split_list:
        result_list.append(_topi.sum(item, axis, True))
    out = _topi.concatenate(tuple(result_list), axis)
    return [out]


_reg.register_schedule("raf.op.tvm.repeat_dx", schedule_generic)


@register_compute("raf.op.tvm.swap_axis")
def swap_axis_compute(attrs, inputs, output_type):
    axis1, axis2 = attrs.axis1, attrs.axis2
    x = inputs[0]
    ndim = len(x.shape)
    axes = list(range(ndim))
    axes[axis1] = axis2
    axes[axis2] = axis1
    out = _topi.transpose(x, axes=axes)
    return [out]


@register_compute("raf.op.tvm.full")
def full_compute(attrs, inputs, output_type):
    out = _topi.full(attrs.shape, attrs.dtype, attrs.fill_value)
    return [out]


@register_compute("raf.op.tvm.full_like")
def full_like_compute(attrs, inputs, output_type):
    out = _topi.full_like(inputs[0], attrs.fill_value)
    return [out]


@register_compute("raf.op.tvm.mesh_grid")
def mesh_grid_compute(attrs, inputs, output_type):
    target_shape = []
    for tensor in inputs:
        target_shape.append(tensor.shape[0])
    out = []

    def fbroadcast(*args):
        return tensor(args[i])

    for i, tensor in enumerate(inputs):
        out.append(_tvm.te.compute(target_shape, fbroadcast))
    return out


@register_compute("raf.op.tvm.scatter_dx")
def scatter_dx_like_compute(attrs, inputs, output_type):
    x, y, dy, index, src = inputs
    for i in range(len(index.shape)):
        # gradient only implement for index and src tensor shape are the same
        assert index.shape[i] == src.shape[i]

    def fcompute(*args):
        return _tvm.tir.if_then_else(x[args] == y[args], dy[args], _tvm.tir.const(0, dy.dtype))

    out = _tvm.te.compute(shape=x.shape, fcompute=fcompute)
    return [out]


_reg.register_strategy("raf.op.tvm.scatter", strategy.scatter_strategy)
_reg.register_injective_schedule("raf.op.tvm.scatter_dx")
_reg.register_injective_schedule("raf.op.tvm.transpose_dx")
_reg.register_injective_schedule("raf.op.tvm.transpose")
_reg.register_injective_schedule("raf.op.tvm.swap_axis")
_reg.register_injective_schedule("raf.op.tvm.mesh_grid")
_reg.register_injective_schedule("raf.op.tvm.split")
_reg.register_injective_schedule("raf.op.tvm.take")
_reg.register_injective_schedule("raf.op.tvm.sequence_mask")
_reg.register_injective_schedule("raf.op.tvm.reverse_sequence")
_reg.register_injective_schedule("raf.op.tvm.concatenate")
_reg.register_injective_schedule("raf.op.tvm.reverse")
_reg.register_injective_schedule("raf.op.tvm.stack")
_reg.register_injective_schedule("raf.op.tvm.squeeze")
_reg.register_injective_schedule("raf.op.tvm.cast")
_reg.register_injective_schedule("raf.op.tvm.cast_like")
_reg.register_injective_schedule("raf.op.tvm.reshape")
_reg.register_broadcast_schedule("raf.op.tvm.broadcast_to")
_reg.register_broadcast_schedule("raf.op.tvm.broadcast_to_like")
_reg.register_broadcast_schedule("raf.op.tvm.clip")
_reg.register_broadcast_schedule("raf.op.tvm.repeat")
_reg.register_broadcast_schedule("raf.op.tvm.expand_dims")
_reg.register_injective_schedule("raf.op.tvm.full")
_reg.register_injective_schedule("raf.op.tvm.full_like")
_reg.register_injective_schedule("raf.op.tvm.batch_flatten")
_reg.register_injective_schedule("raf.op.tvm.arange")
_reg.register_injective_schedule("raf.op.tvm.strided_slice")
_reg.register_reduce_schedule("raf.op.tvm.collapse_sum_like")


@register_compute("raf.op.tvm.take_dx")
def take_dx_compute(attrs, inputs, output_type):
    x, dy, indices = inputs
    axis, mode = int(attrs.axis), attrs.mode
    idim = len(indices.shape)
    # infer axis when negative
    dim = len(x.shape)
    if -dim < axis < 0:
        axis = dim + axis
    if mode == "clip":
        normalized = _topi.minimum(_topi.maximum(indices, 0), x.shape[axis] - 1)
    elif mode == "wrap":
        normalized = _topi.mod(_topi.mod(indices, x.shape[axis]) + x.shape[axis], x.shape[axis])
    else:
        raise ValueError("Not supported mode: " + mode)
    shape = (
        dy.shape[: axis + idim]
        + [
            x.shape[axis],
        ]
        + dy.shape[axis + idim :]
    )
    out1 = _tvm.te.compute(
        shape,
        lambda *idx: _tvm.tir.if_then_else(
            idx[axis + idim] == normalized[idx[axis : axis + idim]],
            dy[idx[: axis + idim] + idx[axis + idim + 1 :]],
            _tvm.tir.const(0, dy.dtype),
        ),
    )
    out2 = _topi.sum(out1, axis=tuple(range(axis, axis + idim))) if idim > 0 else out1
    return [out2]


_reg.register_injective_schedule("raf.op.tvm.take_dx")


@register_compute("raf.op.tvm.strided_slice_dx")
def strided_slice_dx_compute(attrs, inputs, output_type):
    dy = inputs[0]
    begin, end, strides, slice_mode = attrs.begin, attrs.end, attrs.strides, attrs.slice_mode
    var = _tvm.te.placeholder(shape=attrs.primal_shape, dtype=dy.dtype)
    out = _topi.nn.strided_slice(var, begin, end, strides, None, slice_mode)
    grads = _tvm.te.gradient(out, [var], head=dy)
    return grads


_reg.register_injective_schedule("raf.op.tvm.strided_slice_dx")


@register_compute("raf.op.tvm.resize2d")
def compute_resize2d(attrs, inputs, out_type):
    """compute definition for resize2d op"""
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    out_dtype = attrs.out_dtype
    return [
        _topi.image.resize2d(
            inputs[0],
            (None, None, None, None),
            size,
            layout,
            method,
            coord_trans,
            rounding_method,
            bicubic_alpha=cubic_alpha,
            bicubic_exclude=cubic_exclude,
            out_dtype=out_dtype,
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.resize2d")


@register_compute("raf.op.tvm.resize2d_dx")
def resize2d_dx_compute(attrs, inputs, out_type):
    """compute definition for resize2d_dx op"""
    dy = inputs[1]
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    out_dtype = attrs.out_dtype

    out = _topi.image.resize2d(
        inputs[0],
        (None, None, None, None),
        size,
        layout,
        method,
        coord_trans,
        rounding_method,
        bicubic_alpha=cubic_alpha,
        bicubic_exclude=cubic_exclude,
        out_dtype=out_dtype,
    )
    grads = _tvm.te.gradient(out, [inputs[0]], head=dy)
    return grads


_reg.register_injective_schedule("raf.op.tvm.resize2d_dx")


@register_compute("raf.op.tvm.adv_index")
def adv_index_compute(attrs, inputs, output_type):
    data = inputs[0]
    indices = inputs[1:]
    out = _topi.adv_index(data, indices)
    return [out]


_reg.register_injective_schedule("raf.op.tvm.adv_index")


@register_compute("raf.op.tvm.adv_index_dx")
def adv_index_dx_compute(attrs, inputs, output_type):
    # pylint: disable=too-many-locals
    def _get_broadcast_shape(shape1, shape2):
        if shape1 == shape2:
            return shape1
        length1 = len(shape1)
        length2 = len(shape2)
        if length1 > length2:
            shape = list(shape1)
        else:
            shape = list(shape2)
        i = max(length1, length2) - 1
        for a, b in zip(shape1[::-1], shape2[::-1]):
            if a != 1 and b != 1 and a != b:
                raise ValueError("shape1=%s is not broadcastable to shape2=%s" % (shape1, shape2))
            shape[i] = b if a == 1 else a
            i -= 1
        return list(shape)

    dy = inputs[0]
    data = inputs[1]
    indices = inputs[2:]
    idim = len(indices)
    bshape = list(indices[0].shape)
    for ind in indices[1:]:
        bshape = _get_broadcast_shape(bshape, list(ind.shape))

    for i, ind in enumerate(indices):
        if list(ind.shape) != bshape:
            indices[i] = _topi.broadcast_to(ind, bshape)
    shape = bshape + data.shape[:]
    b_len = len(bshape)

    def index_dx(*idx):
        expr = idx[b_len] == indices[0][idx[:b_len]]
        for i in range(1, idim):
            tmp = idx[b_len + i] == indices[i][idx[:b_len]]
            expr = expr & tmp
        return _tvm.tir.if_then_else(
            expr, dy[idx[:b_len] + idx[b_len + idim :]], _tvm.tir.const(0, dy.dtype)
        )

    out1 = _tvm.te.compute(shape, index_dx)
    out2 = _topi.sum(out1, axis=tuple(range(b_len)))
    return [out2]


_reg.register_injective_schedule("raf.op.tvm.adv_index_dx")


@register_compute("raf.op.tvm.clip_dx")
def clip_dx_compute(attrs, inputs, output_type):
    x, dy = inputs
    a_min = _tvm.tir.const(attrs.a_min, x.dtype)
    a_max = _tvm.tir.const(attrs.a_max, x.dtype)

    def _select(*indices):
        return _tvm.tir.if_then_else(
            _tvm.tir.any(x[indices] <= a_min, x[indices] >= a_max), 0, dy(*indices)
        )

    return [_tvm.te.compute(x.shape, _select)]


_reg.register_injective_schedule("raf.op.tvm.clip_dx")


@register_compute("raf.op.tvm.gather_nd_dx")
def gather_nd_dx_compute(attrs, inputs, output_type):
    # pylint: disable=too-many-locals
    data, indices, dy = inputs
    ind_s = _topi.utils.get_const_tuple(indices.shape)
    ind_l = len(ind_s)
    x = ind_s[0]
    ind_s_1 = ind_s[1:]
    data_s = _topi.utils.get_const_tuple(data.shape)
    data_s_0 = data_s[:x]

    def compute_match(*idx):
        ind_i = idx[: ind_l - 1]
        data_i = idx[ind_l - 1 :]
        ret = _tvm.tir.const(True, "bool")
        for i in range(x):
            ind_idx = (i,) + ind_i
            ret = _tvm.tir.And(ret, indices[ind_idx] == data_i[i])
        return ret

    match = _tvm.te.compute(ind_s_1 + data_s_0, compute_match)

    def compute_temp(*idx):
        ind_i = idx[: ind_l - 1]
        data_i_0 = idx[ind_l - 1 : ind_l - 1 + x]
        data_i_1 = idx[ind_l - 1 + x :]
        temp_cond = match[ind_i + data_i_0]
        t_val = dy[ind_i + data_i_1]
        f_val = _tvm.tir.const(0, dy.dtype)
        return _tvm.tir.if_then_else(temp_cond, t_val, f_val)

    temp = _tvm.te.compute(ind_s_1 + data_s, compute_temp)
    ret = _topi.sum(temp, axis=tuple(range(0, ind_l - 1)))
    return [ret]


@register_compute("raf.op.tvm.gather_dx")
def gather_dx_compute(attrs, inputs, output_type):
    data, indices, dy = inputs
    axis = int(attrs.axis)
    dim = len(data.shape)
    if axis < 0:
        assert axis > -dim
        axis = dim + axis
    shape = (
        dy.shape[: axis + 1]
        + [
            data.shape[axis],
        ]
        + dy.shape[axis + 1 :]
    )
    out1 = _tvm.te.compute(
        shape,
        lambda *idx: _tvm.tir.if_then_else(
            idx[axis + 1] == indices[idx[: axis + 1] + idx[axis + 2 :]],
            dy[idx[: axis + 1] + idx[axis + 2 :]],
            _tvm.tir.const(0, dy.dtype),
        ),
    )
    out2 = _topi.sum(out1, axis=axis)
    return [out2]


_reg.register_injective_schedule("raf.op.tvm.gather")
_reg.register_injective_schedule("raf.op.tvm.gather_dx")
_reg.register_injective_schedule("raf.op.tvm.gather_nd")
_reg.register_injective_schedule("raf.op.tvm.gather_nd_dx")


@register_compute("raf.op.tvm.embedding_dx")
def embedding_dx_compute(attrs, inputs, output_type):
    dy, indices = inputs
    num_weight = int(attrs.dims[0])
    idim = len(indices.shape)
    shape = (
        dy.shape[:idim]
        + [
            num_weight,
        ]
        + dy.shape[idim:]
    )
    out1 = _tvm.te.compute(
        shape,
        lambda *idx: _tvm.tir.if_then_else(
            idx[idim] == indices[idx[:idim]],
            dy[idx[:idim] + idx[idim + 1 :]],
            _tvm.tir.const(0, dy.dtype),
        ),
    )
    out2 = _topi.sum(out1, axis=tuple(range(idim))) if idim > 0 else out1
    return [out2]


_reg.register_injective_schedule("raf.op.tvm.embedding_dx")

_reg.register_strategy("raf.op.tvm.cumsum", strategy.cumsum_strategy)
