# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, too-many-locals, unused-argument
"""Reduction compute definition and schedules."""
from operator import mul
from functools import reduce

import numpy as np

from raf._tvm_op.nn import schedule_generic
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .utils import get_cuda_max_thread, profile_schedule

_topi = _tvm.topi  # pylint: disable=invalid-name, no-member


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


def _schedule_cuda_sum_long_reduce(op, sch, **kwargs):
    """The helper function for scheduling sum with long reduction length for CUDA.
    In this case, we want to parallelize the reduction to keep the GPU busy. This is modified
    from TOPI reduce schedule for CUDA.

    Parameters
    ----------
    op: tvm.Operation
        The operator being scheduled.

    sch: tvm.schedule.Schedule
        The working schedule.

    **kwargs: Dict[str, List[Any]]
        Tunable parameters. If not presents, the default values will be used.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # pylint: disable=invalid-name

    # Whether this workload is reducing all axes.
    data_out = op.output(0)
    all_reduce = len(sch[data_out].op.axis) == 0

    # Setup the tunable parameter value.
    num_thread = kwargs.get("num_thread", get_cuda_max_thread() if all_reduce else 32)
    thread_x = _tvm.te.thread_axis((0, num_thread), "threadIdx.x")

    # Fuse and rfactor the reduce axis
    fused_reduce = sch[data_out].fuse(
        *[sch[data_out].op.reduce_axis[i] for i in range(len(sch[data_out].op.reduce_axis))]
    )
    _, ki = sch[data_out].split(fused_reduce, factor=num_thread)
    data_out_rf = sch.rfactor(data_out, ki)
    tx = sch[data_out].op.reduce_axis[0]
    sch[data_out].bind(tx, thread_x)
    sch[data_out_rf].compute_at(sch[data_out], tx)

    if not all_reduce:
        # There are one or more axes to not reduced. Here we bind them to threads and blocks
        # for parallelism.
        block_x = _tvm.te.thread_axis("blockIdx.x")
        thread_y = _tvm.te.thread_axis((0, num_thread), "threadIdx.y")

        # Fuse and split the axis
        fused_outer = sch[data_out].fuse(
            *[sch[data_out].op.axis[i] for i in range(len(sch[data_out].op.axis))]
        )
        bx, outer_in = sch[data_out].split(fused_outer, factor=num_thread)

        # Bind non-reduced axes to threads and blocks
        sch[data_out].bind(outer_in, thread_y)
        sch[data_out].bind(bx, block_x)
        sch[data_out].set_store_predicate(
            _tvm.tir.all(
                thread_x.equal(0), block_x * num_thread + thread_y < reduce(mul, data_out.shape)
            )
        )
    else:
        # All axes are reduced.
        sch[data_out].set_store_predicate(thread_x.equal(0))
    return sch


@profile_schedule(
    num_thread=[16, 32, 64, get_cuda_max_thread()],
    validator=lambda _, reduce_last_axis: not reduce_last_axis,
)
def schedule_cuda_sum_long_reduce(outs, reduce_last_axis, **kwargs):
    """Schedule sum for CUDA. This schedule targets to the sum with long reduction length.
    In this case, we want to parallelize the reduction to keep the GPU busy. This is modified
    from TOPI reduce schedule for CUDA.

    In addition, this schedule is tunable if the last axis is not reduced.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of reduce in the format of an array of tensors.

    reduce_last_axis: bool
        A hint indicating whether the last axis is reduced.

    **kwargs: Dict[str, List[Any]]
        Tunable parameters. If not presents, the default values will be used.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # pylint: disable=unused-argument
    outs = [outs] if isinstance(outs, _tvm.te.tensor.Tensor) else outs
    sch = _tvm.te.create_schedule([x.op for x in outs])
    scheduled_ops = []

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

        for stage in sch.stages:
            if not stage.is_output and isinstance(stage.op, _tvm.te.ComputeOp):
                if is_scheduled(stage) or len(stage.op.reduce_axis) != 0:
                    return False
        return True

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
                _topi.schedule_injective_from_existing(  # pylint: disable=no-member
                    sch, operator.output(0)
                )
            for inp_tensor in operator.input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    if enable_auto_inline:
                        traverse_before_reduce(inp_tensor)
                    else:
                        traverse_after_reduce(inp_tensor)
        elif operator.tag == "comm_reduce":
            if operator not in scheduled_ops:
                _schedule_cuda_sum_long_reduce(operator, sch, **kwargs)
            for inp_tensor in operator.input_tensors:
                if inp_tensor.op not in scheduled_ops:
                    traverse_before_reduce(inp_tensor)
        elif isinstance(operator, _tvm.te.PlaceholderOp):
            pass
        else:
            raise RuntimeError("Unsupported operator tag: %s" % operator.tag)

        scheduled_ops.append(operator)

    for out in outs:
        traverse_after_reduce(out)

    return sch


@profile_schedule(num_thread=[16, 32, 64, get_cuda_max_thread()], max_block=[128, 256, 512])
def schedule_cuda_short_reduce(outs, **kwargs):
    """Schedule sum for CUDA. This schedule targets to the sum with short reduction length.
    In this case, each thread is responsible for reduction. The parallelization is across
    the output elements. This is modified from TOPI injective schedule for CUDA.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of injective in the format of an array of tensors.

    **kwargs: Dict[str, List[Any]]
        Tunable parameters. If not presents, the default values will be used.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # pylint: disable=invalid-name
    # Tunable parameters.
    num_thread = kwargs.get(
        "num_thread", _tvm.target.Target.current(allow_none=False).max_num_threads
    )
    max_block = kwargs.get("max_block", 256)

    def find_nearest_small_factor(num, target):
        """Find the nearest factor of the given number that is smaller than the target."""
        for i in range(target, 0, -1):
            if num % i == 0:
                return i
        # Unreachable because i=1 must hold.
        return -1

    outs = [outs] if isinstance(outs, _tvm.te.tensor.Tensor) else outs
    sch = _tvm.te.create_schedule([x.op for x in outs])

    _tvm.te.schedule.AutoInlineInjective(sch)  # pylint: disable=no-member
    for out in outs:
        if not _topi.utils.is_empty_shape(out.shape):
            fused = sch[out].fuse(*sch[out].op.axis)

            # Vectorize on fp16 data type to enable half2 for better memory bandwidth utilization.
            vector_width = 2 if out.dtype == "float16" else 1

            out_len = _topi.utils.prod(out.shape)

            try:
                const_size = _topi.utils.get_const_int(out_len)

                # Adjust block and thread to make sure they are dividable so that vectorize can be
                # correctly applied.
                if vector_width > 1 and const_size % vector_width == 0:
                    remain_total_size = const_size // vector_width
                    cand_sizes = [0, 0]
                    for idx, max_size in enumerate([num_thread, max_block]):
                        cand_sizes[idx] = (
                            max_size
                            if remain_total_size % max_size == 0
                            else find_nearest_small_factor(remain_total_size, max_size)
                        )
                        remain_total_size //= cand_sizes[idx]

                    # If the product of candidate dividable (block * thread) is too small,
                    # then the performance may be worse even half2 is enabled. Note that 0.7
                    # is just a heuristic ratio and may not be optimal for all workloads.
                    if np.prod(cand_sizes) / (max_block * num_thread) >= 0.7:
                        num_thread, max_block = cand_sizes

                need_block_split = const_size > max_block * num_thread * vector_width
            except ValueError:
                need_block_split = False
                const_size = 0

            if vector_width > 1:
                fused, vec = sch[out].split(fused, vector_width)
                sch[out].vectorize(vec)

            if need_block_split:
                xo, xi = sch[out].split(fused, factor=num_thread * max_block)
                bx, tx = sch[out].split(xi, factor=num_thread)
                sch[out].reorder(bx, tx, xo)
                sch[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
                sch[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
            else:
                if const_size != 0 and const_size < num_thread:
                    bx, tx = sch[out].split(fused, factor=const_size)
                else:
                    bx, tx = sch[out].split(fused, factor=num_thread)
                sch[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
                sch[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
    return sch


@schedule_sum.register(["cuda", "gpu"])
def schedule_sum_cuda(attrs, outs, target):
    # pylint: disable=unused-argument
    def get_num_elements(axes):
        extents = [int(iv.dom.extent) for iv in axes]
        n_elems = 1
        for extent in extents:
            n_elems *= extent
        return n_elems

    def get_sum_input(tensor):
        operator = tensor.op
        if len(operator.input_tensors) != 1:
            return None

        input_tensor = operator.input_tensors[0]
        if isinstance(input_tensor.op, _tvm.te.PlaceholderOp):
            return input_tensor
        return None

    with target:
        out = outs[0]
        num_out_elements = get_num_elements(out.op.axis)
        num_reduce_elements = get_num_elements(out.op.reduce_axis)

        # Whether the last axis is reduced. Note that axis=-1 should already be proceed in advance.
        reduce_last_axis = False
        input_tensor = get_sum_input(out)
        if input_tensor is not None:
            # Only try to analyze the workload with sum as the last op.
            reduce_axis = [False for _ in range(len(input_tensor.shape))]
            for axis in attrs.axis:
                reduce_axis[axis.value] = True
            if attrs.exclude == 1:
                reduce_axis = [not axis for axis in reduce_axis]
            reduce_last_axis = reduce_axis[-1]

        # We attempt to saturate the GPU cores by parallelization, so we dispatch
        # the sum workloads to two schedules based on their reduction length.
        if num_out_elements > num_reduce_elements:
            return schedule_cuda_short_reduce(outs)

        return schedule_cuda_sum_long_reduce(outs, reduce_last_axis=reduce_last_axis)


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
