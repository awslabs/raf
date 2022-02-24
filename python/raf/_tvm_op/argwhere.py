# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Acknowledgement: The main logic originates from TVM

# pylint: disable=missing-function-docstring,invalid-name,undefined-variable
"""Compute definitions and schedules for operator argwhere"""
from .._lib import tvm as _tvm
from .._lib import _reg, _op

_topi = _tvm.topi  # pylint: disable=no-member
_te = _tvm.te  # pylint: disable=no-member


@_te.hybrid.script
def hybrid_argwhere_1d(output_shape, condition):
    """Find the indices of elements of a 1-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        1-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    b = output_tensor((2,), "int64")
    a1 = condition.shape[0]
    valid_index = 0
    for i1 in range(a1):
        if condition[i1] != 0:
            a[valid_index, 0] = i1
            valid_index += 1
    b[0] = int64(valid_index)
    b[1] = int64(1)
    return a, b


@_te.hybrid.script
def hybrid_argwhere_2d(output_shape, condition):
    """Find the indices of elements of a 2-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        2-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    b = output_tensor((2,), "int64")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            if condition[i1, i2] != 0:
                a[valid_index, 0] = i1
                a[valid_index, 1] = i2
                valid_index += 1
    b[0] = int64(valid_index)
    b[1] = int64(2)
    return a, b


@_te.hybrid.script
def hybrid_argwhere_3d(output_shape, condition):
    """Find the indices of elements of a 3-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        3-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    b = output_tensor((2,), "int64")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            for i3 in range(a3):
                if condition[i1, i2, i3] != 0:
                    a[valid_index, 0] = i1
                    a[valid_index, 1] = i2
                    a[valid_index, 2] = i3
                    valid_index += 1
    b[0] = int64(valid_index)
    b[1] = int64(3)
    return a, b


@_te.hybrid.script
def hybrid_argwhere_4d(output_shape, condition):
    """Find the indices of elements of a 4-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        4-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    b = output_tensor((2,), "int64")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    a4 = condition.shape[3]
    valid_index = 0
    for i1 in range(a1):
        for i2 in range(a2):
            for i3 in range(a3):
                for i4 in range(a4):
                    if condition[i1, i2, i3, i4] != 0:
                        a[valid_index, 0] = i1
                        a[valid_index, 1] = i2
                        a[valid_index, 2] = i3
                        a[valid_index, 3] = i4
                        valid_index += 1
    b[0] = int64(valid_index)
    b[1] = int64(4)
    return a, b


@_te.hybrid.script
def hybrid_argwhere_5d(output_shape, condition):
    """Find the indices of elements of a 5-D tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        5-D tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    a = output_tensor(output_shape, "int32")
    b = output_tensor((2,), "int64")
    a1 = condition.shape[0]
    a2 = condition.shape[1]
    a3 = condition.shape[2]
    a4 = condition.shape[3]
    a5 = condition.shape[4]
    valid_index = 0
    for i1 in range(a1):  # pylint: disable=too-many-nested-blocks
        for i2 in range(a2):
            for i3 in range(a3):
                for i4 in range(a4):
                    for i5 in range(a5):
                        if condition[i1, i2, i3, i4, i5] != 0:
                            a[valid_index, 0] = i1
                            a[valid_index, 1] = i2
                            a[valid_index, 2] = i3
                            a[valid_index, 3] = i4
                            a[valid_index, 4] = i5
                            valid_index += 1
    b[0] = int64(valid_index)
    b[1] = int64(5)
    return a, b


def argwhere_cpu(output_shape, condition):
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    condition : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    if len(condition.shape) == 1:
        return hybrid_argwhere_1d(output_shape.shape, condition)
    if len(condition.shape) == 2:
        return hybrid_argwhere_2d(output_shape.shape, condition)
    if len(condition.shape) == 3:
        return hybrid_argwhere_3d(output_shape.shape, condition)
    if len(condition.shape) == 4:
        return hybrid_argwhere_4d(output_shape.shape, condition)
    if len(condition.shape) == 5:
        return hybrid_argwhere_5d(output_shape.shape, condition)
    raise ValueError("Does not support rank higher than 5 in argwhere")


def wrap_compute_argwhere(topi_compute):
    """wrap argwhere topi compute"""

    def _compute_argwhere(attrs, inputs, out_type):  # pylint: disable=unused-argument
        output_shape = []
        for s in out_type.fields[0].shape:
            if hasattr(s, "value"):
                output_shape.append(s)
            else:
                output_shape.append(_te.var("any_dim", "int32"))
        new_output_type = _tvm.ir.TensorType(output_shape, "int32")
        return topi_compute(new_output_type, inputs[0])

    return _compute_argwhere


@_tvm.target.override_native_generic_func("raf_argwhere_strategy")
def raf_argwhere_strategy(attrs, inputs, out_type, target):  # pylint: disable=unused-argument
    """argwhere generic strategy"""
    strategy = _op.op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argwhere(argwhere_cpu),
        _op.strategy.generic.wrap_topi_schedule(_topi.generic.schedule_argwhere),
        name="raf.argwhere.generic",
    )
    return strategy


fdiv = _tvm.tir.floordiv
fmod = _tvm.tir.floormod


def compact_nonzero_indices_ir(condition, write_indices, out, out_shape, do_write_func):
    """Copy nonzero indices to the corresponding write locations.

    Parameters
    ----------
    condition : Buffer
        The input condition.

    write_indices : Buffer
        The result of exclusive scan on a boolean array, where True indicates that
        the condition is non zero at that position.

    out : Buffer
        The output buffer to copy indices to.

    do_write_func : a function
        A callback that accepts an output buffer, a dst index to write to, and a src index.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    ib = _tvm.tir.ir_builder.create()
    ndim = _tvm.tir.const(len(condition.shape), "int64")
    size_1d = _topi.utils.prod(condition.shape)

    condition = ib.buffer_ptr(condition)
    write_indices = ib.buffer_ptr(write_indices)
    out = ib.buffer_ptr(out)
    out_shape = ib.buffer_ptr(out_shape)

    nthread_tx = int(_tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_bx = _topi.utils.ceil_div(size_1d, nthread_tx)
    tx = _te.thread_axis("threadIdx.x")
    bx = _te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)

    with ib.new_scope():
        idx = bx * nthread_tx + tx
        with ib.if_scope(idx < size_1d):
            with ib.if_scope(condition[idx] != 0):
                do_write_func(out, write_indices[idx], idx)
        with ib.if_scope(idx == 0):
            valid_index = _topi.cast(write_indices[size_1d - 1], dtype="int64")
            valid_index += _tvm.tir.if_then_else(
                condition[size_1d - 1] == 0, _tvm.tir.const(0, "int64"), _tvm.tir.const(1, "int64")
            )
            out_shape[0] = valid_index
        with ib.if_scope(idx == 1):
            out_shape[1] = ndim

    return ib.get()


def argwhere_common(output_shape, condition, do_write_func):
    """A common compute used by argwhere of various ranks.

    Parameters
    ----------
    output_shape : list of int or tvm.tir.Any
        Tensor with output shape info.

    condition : tvm.te.Tensor
        The input condition.

    do_write_func : a function
        A callback that accepts an output buffer, a dst index to write to, and a src index.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """

    flags = _topi.not_equal(condition, _tvm.tir.const(0))
    flags_1d = _topi.reshape(flags, (_topi.utils.prod(flags.shape),))
    write_indices = _topi.cuda.exclusive_scan(_topi.cast(flags_1d, dtype="int32"))

    condition_buf = _tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    write_indices_buf = _tvm.tir.decl_buffer(
        write_indices.shape, write_indices.dtype, "write_indices_buf", data_alignment=8
    )
    out_buf = _tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)
    out_shape_buf = _tvm.tir.decl_buffer((2,), "int64", "out_buf", data_alignment=8)

    out = _te.extern(
        [output_shape, (2,)],
        [condition, write_indices],
        lambda ins, outs: compact_nonzero_indices_ir(
            ins[0], ins[1], outs[0], outs[1], do_write_func
        ),
        dtype=["int32"],
        in_buffers=[condition_buf, write_indices_buf],
        out_buffers=[out_buf, out_shape_buf],
        name="argwhere",
        tag="argwhere_gpu",
    )

    return out


def argwhere_1d(output_shape, condition):
    """Compute for argwhere 1D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def do_write(out, write_index, idx):
        out[write_index] = idx

    return argwhere_common(output_shape, condition, do_write)


def argwhere_2d(output_shape, condition):
    """Compute for argwhere 2D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def do_write(out, write_index, idx):
        a1 = condition.shape[1]
        out[write_index * 2] = fdiv(idx, a1)
        out[write_index * 2 + 1] = fmod(idx, a1)

    return argwhere_common(output_shape, condition, do_write)


def argwhere_3d(output_shape, condition):
    """Compute for argwhere 3D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def do_write(out, write_index, idx):
        _, a1, a2 = condition.shape
        s1 = a1 * a2
        out[write_index * 3] = fdiv(idx, s1)
        out[write_index * 3 + 1] = fdiv(fmod(idx, s1), a2)
        out[write_index * 3 + 2] = fmod(idx, a2)

    return argwhere_common(output_shape, condition, do_write)


def argwhere_4d(output_shape, condition):
    """Compute for argwhere 4D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def do_write(out, write_index, idx):
        _, a1, a2, a3 = condition.shape
        s1 = a2 * a3
        s2 = a1 * s1
        out[write_index * 4] = fdiv(idx, s2)
        out[write_index * 4 + 1] = fdiv(fmod(idx, s2), s1)
        out[write_index * 4 + 2] = fdiv(fmod(idx, s1), a3)
        out[write_index * 4 + 3] = fmod(idx, a3)

    return argwhere_common(output_shape, condition, do_write)


def argwhere_5d(output_shape, condition):
    """Compute for argwhere 5D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def do_write(out, write_index, idx):
        _, a1, a2, a3, a4 = condition.shape
        s1 = a3 * a4
        s2 = a2 * s1
        s3 = a1 * s2
        out[write_index * 5] = fdiv(idx, s3)
        out[write_index * 5 + 1] = fdiv(fmod(idx, s3), s2)
        out[write_index * 5 + 2] = fdiv(fmod(idx, s2), s1)
        out[write_index * 5 + 3] = fdiv(fmod(idx, s1), a4)
        out[write_index * 5 + 4] = fmod(idx, a4)

    return argwhere_common(output_shape, condition, do_write)


def argwhere_gpu(output_shape, condition):
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    output_shape : tvm.te.Tensor
        Tensor with output shape info.

    condition : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    if len(condition.shape) == 1:
        return argwhere_1d(output_shape.shape, condition)
    if len(condition.shape) == 2:
        return argwhere_2d(output_shape.shape, condition)
    if len(condition.shape) == 3:
        return argwhere_3d(output_shape.shape, condition)
    if len(condition.shape) == 4:
        return argwhere_4d(output_shape.shape, condition)
    if len(condition.shape) == 5:
        return argwhere_5d(output_shape.shape, condition)
    raise ValueError("Argwhere does not support rank higher than 5")


def schedule_argwhere_gpu(outs):
    """Schedule for argwhere on cuda.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argwhere
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for argwhere
    """
    outs = [outs] if isinstance(outs, _te.tensor.Tensor) else outs
    s = _te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if _topi.tag.is_injective(op.tag):
            _topi.cuda.injective.schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for out in outs:
        traverse(out.op)
    return s


@raf_argwhere_strategy.register(["cuda", "gpu"])
def raf_argwhere_strategy_cuda(attrs, inputs, out_type, target):  # pylint: disable=unused-argument
    """argwhere cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argwhere(argwhere_gpu),
        _op.strategy.generic.wrap_topi_schedule(schedule_argwhere_gpu),
        name="raf.argwhere.cuda",
    )
    return strategy


_reg.register_strategy("raf.op.tvm.upper_bound.argwhere", raf_argwhere_strategy)
