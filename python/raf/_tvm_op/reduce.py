# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, too-many-locals, unused-argument
"""Reduction compute definition and schedules."""
from operator import mul
from functools import reduce

from raf._tvm_op.nn import schedule_generic
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .utils import profile_schedule

_topi = _tvm.topi  # pylint: disable=invalid-name, no-member


def _schedule_cuda_reduce(op, sch, is_idx_reduce=False, **kwargs):
    # Setup the tunable parameter value.
    num_thread = kwargs.get("num_thread", 32)

    if is_idx_reduce:
        data_out = op.input_tensors[0]
    else:
        data_out = op.output(0)

    if not sch[data_out].op.reduce_axis:
        return _topi.cuda.schedule_injective_from_existing(sch, op.output(0))

    if len(sch[data_out].op.axis) > 0:
        all_reduce = False
        block_x = _tvm.te.thread_axis("blockIdx.x")
        thread_x = _tvm.te.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = _tvm.te.thread_axis((0, num_thread), "threadIdx.y")
    else:
        all_reduce = True
        num_thread = _tvm.target.Target.current(allow_none=False).max_num_threads
        thread_x = _tvm.te.thread_axis((0, num_thread), "threadIdx.x")

    # Fuse and refactor the reduce axis
    fused_reduce = sch[data_out].fuse(
        *[sch[data_out].op.reduce_axis[i] for i in range(len(sch[data_out].op.reduce_axis))]
    )
    ko, ki = sch[data_out].split(fused_reduce, factor=num_thread)
    if is_idx_reduce:
        data_out_rf, _ = sch.rfactor(data_out, ki)
    else:
        data_out_rf = sch.rfactor(data_out, ki)
    tx = sch[data_out].op.reduce_axis[0]
    sch[data_out].bind(tx, thread_x)
    sch[data_out_rf].compute_at(sch[data_out], tx)
    if is_idx_reduce:
        real_output = op.output(0)
        temp_idx_input = data_out.op.output(0)
        temp_val_input = data_out.op.output(1)
    else:
        real_output = data_out
    if not all_reduce:
        # Fuse and split the axis
        fused_outer = sch[real_output].fuse(
            *[sch[real_output].op.axis[i] for i in range(len(sch[real_output].op.axis))]
        )
        bx, outer_in = sch[real_output].split(fused_outer, factor=num_thread)

        # Bind the axes to threads and blocks
        sch[real_output].bind(outer_in, thread_y)
        sch[real_output].bind(bx, block_x)
        if is_idx_reduce:
            sch[temp_idx_input].compute_at(sch[real_output], outer_in)
            sch[temp_val_input].compute_at(sch[real_output], outer_in)
        sch[real_output].set_store_predicate(
            _tvm.tir.all(
                thread_x.equal(0), block_x * num_thread + thread_y < reduce(mul, real_output.shape)
            )
        )
    else:
        if is_idx_reduce:
            spatial_axis = sch[real_output].fuse(*(sch[real_output].op.axis))
            sch[real_output].bind(spatial_axis, _tvm.te.thread_axis("blockIdx.x"))
            sch[temp_idx_input].compute_at(sch[real_output], spatial_axis)
            sch[temp_val_input].compute_at(sch[real_output], spatial_axis)
        sch[real_output].set_store_predicate(thread_x.equal(0))
    return sch


def _enable_auto_inline(sch):
    def is_scheduled(stage):
        # auto inline requires the attach type is AttachType.kGroupRoot
        conds = [
            len(stage.relations) == 0,
            stage.attach_type == 1,
            stage.all_iter_vars == stage.leaf_iter_vars,
        ]
        if not all(conds):
            return True
        return False

    for s in sch.stages:
        if not s.is_output and isinstance(s.op, _tvm.te.ComputeOp):
            if is_scheduled(s) or len(s.op.reduce_axis) != 0:
                return False
    return True


@profile_schedule(num_thread=[8, 16, 32, 64], tt=[False, True])
def schedule_cuda_reduce(outs, **kwargs):
    """Schedule for inject->reduce->bcast ops.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of reduce in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, _tvm.te.tensor.Tensor) else outs
    sch = _tvm.te.create_schedule([x.op for x in outs])
    scheduled_ops = []
    enable_auto_inline = _enable_auto_inline(sch)

    def traverse_before_reduce(tensor):
        """Internal traverse function"""
        operator = tensor.op
        if isinstance(operator, _tvm.te.PlaceholderOp):
            return
        if _topi.tag.is_injective(operator.tag):
            sch[operator].compute_inline()
            for inp_tensor in operator.input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    traverse_before_reduce(inp_tensor)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    def traverse_after_reduce(tensor):
        """Internal traverse function"""
        operator = tensor.op
        if _topi.tag.is_broadcast(operator.tag):
            if operator not in scheduled_ops:
                _topi.schedule_injective_from_existing(sch, operator.output(0))
            for inp_tensor in operator.input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    if enable_auto_inline:
                        traverse_before_reduce(inp_tensor)
                    else:
                        traverse_after_reduce(inp_tensor)
        elif operator.tag == "comm_reduce":
            if operator not in scheduled_ops:
                _schedule_cuda_reduce(operator, sch, is_idx_reduce=False, **kwargs)
            for inp_tensor in operator.input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    traverse_before_reduce(inp_tensor)
        elif operator.tag == "comm_reduce_idx":
            if operator not in scheduled_ops:
                _schedule_cuda_reduce(operator, sch, is_idx_reduce=True, **kwargs)
            input_tensors = operator.input_tensors[0].op.input_tensors
            for inp_tensor in input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    traverse_before_reduce(inp_tensor)
        elif isinstance(operator, _tvm.te.PlaceholderOp):
            pass
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    for out in outs:
        traverse_after_reduce(out)

    return sch


@register_compute("raf.op.tvm.sum")
def sum_compute(attrs, inputs, output_type):  # pylint: disable=no-member
    x = inputs[0]
    axes = list(_topi.utils.get_const_tuple(attrs.axis))
    keep = list(_topi.utils.get_const_tuple(attrs.keepdims))
    exclude = attrs.exclude
    if exclude:
        axes = axis_exclude(axes, x.shape)
    if not keep:
        # TODO(@were): It seems that TVM create view may crash, I cannot directly return [x]
        return [
            _tvm.te.compute(x.shape, lambda *args: x(*args))  # pylint: disable=unnecessary-lambda
        ]
    if len(keep) == 1:
        axes = None if not axes else axes
        return [_topi.sum(x, axis=axes, keepdims=keep[0])]
    axes = sorted(zip(axes, keep))
    red_axis = [
        _tvm.te.reduce_axis((0, x.shape[i]), name="rv%d" % e) for e, (i, _) in enumerate(axes)
    ]
    shape = list(x.shape)
    for i, j in axes:
        shape[i] = 1 if j else None
    shape = [i for i in shape if i is not None]

    def fcompute(*args):
        scan = list(args[::-1])
        reds = red_axis[::-1]
        idx = []
        for i in range(len(x.shape)):
            if (i, True) in axes:
                idx.append(reds.pop())
                scan.pop()
            elif (i, False) in axes:
                idx.append(reds.pop())
            else:
                idx.append(scan.pop())
        return _tvm.te.sum(x(*idx), axis=red_axis)

    return [_tvm.te.compute(shape=shape, fcompute=fcompute, tag="comm_reduce")]


@generic_func
def schedule_sum(attrs, outs, target):
    # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_sum.register(["cuda", "gpu"])
def schedule_sum_cuda(attrs, outs, target):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    def get_num_elements(axes):
        extents = [int(iv.dom.extent) for iv in axes]
        n_elems = 1
        for extent in extents:
            n_elems *= extent
        return n_elems

    with target:
        out = outs[0]
        num_out_elements = get_num_elements(out.op.axis)
        num_reduce_elements = get_num_elements(out.op.reduce_axis)
        sum_last_dim = len(attrs.axis) == 1 and attrs.axis[0] == len(out.shape) - 1

        # We want to saturate the GPU cores by parallelization. There are 2 scenarios
        # 1) Reduce dimension is small - In this case, each thread is responsible for reduction.
        # The parallelization is across the output elements.
        # 2) Reduce dimension is large - We want to parallelize the reduction to keep the GPU busy.
        # Here we fall back to TVM schedule.
        if num_out_elements > num_reduce_elements and sum_last_dim:
            return _topi.cuda.schedule_injective(outs)

        return schedule_cuda_reduce(outs)


_reg.register_schedule("raf.op.tvm.sum", schedule_sum)
_reg.register_reduce_schedule("raf.op.tvm.argmax")
_reg.register_reduce_schedule("raf.op.tvm.argmin")
_reg.register_reduce_schedule("raf.op.tvm.max")
_reg.register_reduce_schedule("raf.op.tvm.min")
_reg.register_reduce_schedule("raf.op.tvm.all")
_reg.register_reduce_schedule("raf.op.tvm.any")
_reg.register_reduce_schedule("raf.op.tvm.mean")
_reg.register_reduce_schedule("raf.op.tvm.prod")


def axis_reverse(input_axis):
    # get the reverse axis to change axis back
    # e.g.: input_axis = [1, 2, 0, 3]
    #       the reverse_axis = [2, 0, 1, 3]
    reverse_axis = input_axis.copy()
    for i, axis in enumerate(input_axis):
        reverse_axis[axis] = i
    return reverse_axis


def axis_exclude(input_axis, shape):
    # get the excluded axis which is not in the input axis
    # e.g.: input_axis = [1, 3] with a total of 4 dimension
    #       the reverse_axis = [0,2]
    total_dim = len(shape)
    exclude_axis = list()
    for i in range(total_dim):
        if i not in input_axis:
            exclude_axis.append(i)
    return exclude_axis


def mul_shapes(shape_list, axis_list):
    # get the product of shapes in given axis
    out = 1
    for axis in axis_list:
        out *= shape_list[axis]
    return out


@register_compute("raf.op.tvm.prod_dx")
def prod_dx_compute(attrs, inputs, output_type):
    x, dy = inputs
    axis = list(_topi.utils.get_const_tuple(attrs.axis))
    exclude = attrs.exclude
    if exclude:
        axis = axis_exclude(axis, x.shape)
    ndim = len(x.shape)
    shape = x.shape
    output_dim = []
    reduce_dim = []
    reduce_axis = []
    for dim in range(ndim):
        if dim not in axis:
            output_dim.append(shape[dim])
            reduce_dim.append(shape[dim])
        else:
            output_dim.append(1)
            reduce_axis.append(_tvm.te.reduce_axis((0, shape[dim]), name="rv%d" % dim))

    product = _tvm.te.comm_reducer(lambda x, y: x * y, lambda t: _tvm.tir.const(1, dtype=t))

    def fcompute(*args):
        args = list(args)
        for i, dim in enumerate(axis):
            args.insert(dim, reduce_axis[i])
        return product(x(*args), axis=reduce_axis)

    prod_x = _tvm.te.compute(reduce_dim, fcompute, name="prod_x")
    dy_reshape = _topi.reshape(dy, output_dim)
    prod_x_reshape = _topi.reshape(prod_x, output_dim)
    factor = _topi.divide(prod_x_reshape, x)
    out = _topi.multiply(factor, dy_reshape)
    return [out]


_reg.register_schedule("raf.op.tvm.prod_dx", schedule_generic)


@register_compute("raf.op.tvm.mean_dx")
def mean_dx_compute(attrs, inputs, output_type):
    dy = inputs[0]
    axis = sorted(list(_topi.utils.get_const_tuple(attrs.axis)))
    keepdims = attrs.keepdims
    exclude = attrs.exclude
    shape = list(_topi.utils.get_const_tuple(attrs.shape))
    ndim = len(shape)
    expandshape = list()
    if exclude:
        axis = axis_exclude(axis, shape)
    shape_mul = mul_shapes(shape, axis)
    for dim in range(ndim):
        if dim not in axis:
            expandshape.append(shape[dim])
        else:
            expandshape.append(1)

    def _elem_div(*indices):
        return dy[indices] / shape_mul

    out = _tvm.te.compute(dy.shape, _elem_div)

    def fbroadcast(*args):
        args = list(args)
        for dim in axis[::-1]:
            del args[dim]
        return out(*args)

    def fbroadcast_keepdim(*args):
        args = list(args)
        for dim in axis:
            args[dim] = 0
        return out(*args)

    if keepdims:
        out_broadcast = _tvm.te.compute(shape, fbroadcast_keepdim)
    else:
        out_broadcast = _tvm.te.compute(shape, fbroadcast)

    return [out_broadcast]


_reg.register_injective_schedule("raf.op.tvm.mean_dx")


@register_compute("raf.op.tvm.sum_dx")
def sum_dx_compute(attrs, inputs, output_type):  # pylint: disable=no-member
    x, dy = inputs
    axes = list(_topi.utils.get_const_tuple(attrs.axis))
    exclude = attrs.exclude
    keepdims = list(_topi.utils.get_const_tuple(attrs.keepdims))
    if exclude:
        axes = axis_exclude(axes, x.shape)
    naxes = len(axes)
    shape = x.shape
    ndim = len(shape)
    if len(keepdims) != naxes and len(keepdims) == 1:
        keepdims = keepdims * naxes
    if len(keepdims) == 0 and len(axes) == 0:
        axes = list(range(ndim))
        keepdims = [0] * ndim

    def fbroadcast(*args):
        args = list(args)
        for i, dim in enumerate(axes[::-1]):
            if keepdims[naxes - 1 - i]:
                args[dim] = 0
            else:
                del args[dim]
        return dy(*args)

    out_broadcast = _tvm.te.compute(shape, fbroadcast)
    return [out_broadcast]


_reg.register_injective_schedule("raf.op.tvm.sum_dx")


@register_compute("raf.op.tvm.l2norm")
def l2norm_compute(attrs, inputs, output_type):  # pylint: disable=no-member
    x = inputs[0]
    res = _topi.multiply(x, x)
    res = _topi.sum(res, axis=[], keepdims=0)
    res = _topi.sqrt(res)
    return [res]


_reg.register_reduce_schedule("raf.op.tvm.l2norm")
