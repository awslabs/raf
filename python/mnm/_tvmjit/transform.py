# pylint: disable=missing-function-docstring, line-too-long, undefined-loop-variable
"""Compute definition and schedules for data transform operators"""
from mnm._tvmjit.nn import schedule_layer_norm
from .._lib import register_compute
from .._lib import strategy
from .._lib import tvm as _tvm  # pylint: disable=unused-import
from .._lib import _reg
_topi = _tvm.topi  # pylint: disable=invalid-name,no-member

@register_compute("mnm.op.transpose_dx")
def transpose_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    dy = inputs[0]
    axes = list(_topi.utils.get_const_tuple(attrs.axes))
    axes_inverse = axes.copy()
    for idx, i in enumerate(axes):
        axes_inverse[i] = idx
    out = _topi.transpose(dy, axes=tuple(axes_inverse))
    return [out]


@register_compute("mnm.op.repeat_dx")
def repeat_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
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

_reg.register_schedule("mnm.op.repeat_dx", schedule_layer_norm)

@register_compute("mnm.op.swap_axis")
def swap_axis_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    axis1, axis2 = attrs.axis1, attrs.axis2
    x = inputs[0]
    ndim = len(x.shape)
    axes = list(range(ndim))
    axes[axis1] = axis2
    axes[axis2] = axis1
    out = _topi.transpose(x, axes=axes)
    return [out]

@register_compute("mnm.op.full")
def full_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    out = _topi.full(attrs.shape, attrs.dtype, attrs.fill_value)
    return [out]

@register_compute("mnm.op.full_like")
def full_like_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    out = _topi.full_like(inputs[0], attrs.fill_value)
    return [out]

@register_compute("mnm.op.mesh_grid")
def mesh_grid_compute(attrs, inputs, output_type): # pylint: disable=unused-argument
    target_shape = []
    for tensor in inputs:
        target_shape.append(tensor.shape[0])
    out = []
    def fbroadcast(*args):
        return tensor(args[i])

    for i, tensor in enumerate(inputs):
        out.append(_tvm.te.compute(target_shape, fbroadcast))
    return out

@register_compute("mnm.op.scatter_dx")
def scatter_dx_like_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument, line-too-long
    x = inputs[0]
    y = inputs[1]
    dy = inputs[2]
    index = inputs[3]
    src = inputs[4]
    for i in range(len(index.shape)):
        #gradient only implement for index and src tensor shape are the same
        assert index.shape[i] == src.shape[i]

    def fcompute(*args):
        return _tvm.tir.if_then_else(x[args] == y[args], dy[args], _tvm.tir.const(0, dy.dtype))
    out = _tvm.te.compute(shape=x.shape, fcompute=fcompute)
    return [out]


_reg.register_strategy("mnm.op.scatter", strategy.scatter_strategy)
_reg.register_injective_schedule("mnm.op.scatter_dx")
_reg.register_injective_schedule("mnm.op.transpose_dx")
_reg.register_injective_schedule("mnm.op.transpose")
_reg.register_injective_schedule("mnm.op.swap_axis")
_reg.register_injective_schedule("mnm.op.mesh_grid")
_reg.register_injective_schedule("mnm.op.split")
_reg.register_injective_schedule("mnm.op.take")
_reg.register_injective_schedule("mnm.op.sequence_mask")
_reg.register_injective_schedule("mnm.op.reverse_sequence")
_reg.register_injective_schedule("mnm.op.concatenate")
_reg.register_injective_schedule("mnm.op.reverse")
_reg.register_injective_schedule("mnm.op.stack")
_reg.register_injective_schedule("mnm.op.squeeze")
_reg.register_injective_schedule("mnm.op.cast")
_reg.register_injective_schedule("mnm.op.cast_like")
_reg.register_injective_schedule("mnm.op.reshape")
_reg.register_broadcast_schedule("mnm.op.broadcast_to")
_reg.register_broadcast_schedule("mnm.op.broadcast_to_like")
_reg.register_broadcast_schedule("mnm.op.clip")
_reg.register_broadcast_schedule("mnm.op.repeat")
_reg.register_broadcast_schedule("mnm.op.expand_dims")
_reg.register_injective_schedule("mnm.op.full")
_reg.register_injective_schedule("mnm.op.full_like")
_reg.register_injective_schedule("mnm.op.batch_flatten")
_reg.register_injective_schedule("mnm.op.arange")

@register_compute("mnm.op.adv_index")
def adv_index_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    data = inputs[0]
    indices = inputs[1:]
    out = _topi.adv_index(data, indices)
    return [out]


_reg.register_injective_schedule("mnm.op.adv_index")


@register_compute("mnm.op.adv_index_dx")
def adv_index_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    dy = inputs[0]
    data = inputs[1]
    indices = inputs[2:]
    idim = len(indices)
    bshape = list(indices[0].shape)
    for ind in indices[1:]:
        bshape = max(bshape, list(ind.shape))

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
        return _tvm.tir.if_then_else(expr, dy[idx[:b_len] + idx[b_len + idim:]],
                                     _tvm.tir.const(0, dy.dtype))

    A = _tvm.te.compute(shape, index_dx)
    B = _topi.sum(A, axis=tuple(range(b_len)))
    return [B]

_reg.register_injective_schedule("mnm.op.adv_index_dx")


@register_compute("mnm.op.clip_dx")
def clip_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    x = inputs[0]
    grad = inputs[1]
    a_min = attrs.a_min
    a_max = attrs.a_max

    def _select(*indices):
        return _tvm.tir.if_then_else(_tvm.tir.any(x[indices] <= a_min,
                                                  x[indices] >= a_max),
                                     0, grad(*indices))
    return [_tvm.te.compute(x.shape, _select)]


_reg.register_injective_schedule("mnm.op.clip_dx")

# pylint: disable=too-many-locals
@register_compute("mnm.op.gather_nd_dx")
def gather_nd_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    data, indices, dy = inputs
    ind_s = _topi.utils.get_const_tuple(indices.shape)
    ind_l = len(ind_s)
    x = ind_s[0]
    ind_s_1 = ind_s[1:]
    data_s = _topi.utils.get_const_tuple(data.shape)
    data_s_0 = data_s[:x]
    def compute_match(*idx):
        ind_i = idx[:ind_l - 1]
        data_i = idx[ind_l - 1:]
        ret = _tvm.tir.const(True, "bool")
        for i in range(x):
            ind_idx = (i,) + ind_i
            ret = _tvm.tir.And(ret, indices[ind_idx] == data_i[i])
        return ret
    match = _tvm.te.compute(ind_s_1 + data_s_0, compute_match)
    def compute_temp(*idx):
        ind_i = idx[:ind_l - 1]
        data_i_0 = idx[ind_l - 1: ind_l - 1 + x]
        data_i_1 = idx[ind_l - 1 + x:]
        temp_cond = match[ind_i + data_i_0]
        t_val = dy[ind_i + data_i_1]
        f_val = _tvm.tir.const(0, dy.dtype)
        return _tvm.tir.if_then_else(temp_cond, t_val, f_val)
    temp = _tvm.te.compute(ind_s_1 + data_s, compute_temp)
    ret = _topi.sum(temp, axis=tuple(range(0, ind_l - 1)))
    return [ret]

@register_compute("mnm.op.gather_dx")
def gather_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    data, indices, dy = inputs
    axis = int(attrs.axis)
    dim = len(data.shape)
    if axis < 0:
        assert axis > -dim
        axis = dim + axis
    shape = dy.shape[:axis+1] + [data.shape[axis],] + dy.shape[axis + 1:]
    A = _tvm.te.compute(shape, lambda *idx:
                        _tvm.tir.if_then_else(idx[axis + 1] ==
                                              indices[idx[: axis + 1] + idx[axis + 2:]],
                                              dy[idx[: axis + 1] + idx[axis + 2:]],
                                              _tvm.tir.const(0, dy.dtype)))
    B = _topi.sum(A, axis=axis)
    return [B]

_reg.register_injective_schedule("mnm.op.gather")
_reg.register_injective_schedule("mnm.op.gather_dx")
_reg.register_injective_schedule("mnm.op.gather_nd")
_reg.register_injective_schedule("mnm.op.gather_nd_dx")

@register_compute("mnm.op.where_dx")
def where_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    x1, x2, y, dy = inputs
    def _where_x(x):
        def _f(*idx):
            return _tvm.tir.if_then_else(y[idx] == x[idx], dy[idx], _tvm.tir.const(0, dy.dtype))
        return _f

    assert x1.shape[:] == x2.shape[:] or len(x1.shape) == 0 or len(x2.shape) == 0
    if x1.shape[:] == x2.shape[:]:
        dx1 = _tvm.te.compute(x1.shape, _where_x(x1))
        dx2 = _tvm.te.compute(x1.shape, _where_x(x2))
    elif len(x2.shape) == 0:
        x2 = _topi.broadcast_to(x2, x1.shape)
        dx1 = _tvm.te.compute(x1.shape, _where_x(x1))
        dx2 = _tvm.te.compute(x2.shape, _where_x(x2))
        dx2 = _topi.sum(dx2, axis=tuple(range(len(x2.shape))))
    else:
        x1 = _topi.broadcast_to(x1, x2.shape)
        dx1 = _tvm.te.compute(x1.shape, _where_x(x1))
        dx1 = _topi.sum(dx1, axis=tuple(range(len(x1.shape))))
        dx2 = _tvm.te.compute(x2.shape, _where_x(x2))
    return [dx1, dx2]

_reg.register_broadcast_schedule("mnm.op.where_dx")
