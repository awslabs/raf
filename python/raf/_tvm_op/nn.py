# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, missing-module-docstring
# pylint: disable=unused-argument, invalid-name, too-many-statements
from functools import reduce
import operator

from . import cuda
from .._lib import register_compute
from .._lib import generic_func
from .._lib import tvm as _tvm
from .._lib import _reg
from .._lib import strategy
from .._lib import random

_topi = _tvm.topi  # pylint: disable=no-member

_reg.register_injective_schedule("raf.op.tvm.pad")

_reg.register_strategy("raf.op.tvm.dense", strategy.dense_strategy)


def compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False):
    if len(inputs) == 2:
        data, weight = inputs[0], inputs[1]
    else:
        raise ValueError("Invalid input")
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    return [_topi.matmul(data, weight, transp_a=transpose_a, transp_b=transpose_b)]


@register_compute("raf.op.tvm.matmul")
def compute_matmul(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False)


@register_compute("raf.op.tvm.matmul_tn")
def compute_matmul_tn(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=False)


@register_compute("raf.op.tvm.matmul_nt")
def compute_matmul_nt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=True)


@register_compute("raf.op.tvm.matmul_tt")
def compute_matmul_tt(attr, inputs, output_type):
    return compute_matmul_general(attr, inputs, output_type, transpose_a=True, transpose_b=True)


_reg.register_injective_schedule("raf.op.tvm.matmul")
_reg.register_injective_schedule("raf.op.tvm.matmul_tn")
_reg.register_injective_schedule("raf.op.tvm.matmul_nt")
_reg.register_injective_schedule("raf.op.tvm.matmul_tt")


def compute_batch_matmul_general(attr, inputs, output_type, transpose_a=False, transpose_b=False):
    assert len(inputs) == 2, "Expected 2 inputs, but got {}".format(len(inputs))
    data, weight = inputs[0], inputs[1]
    assert len(data.shape) == 3 and len(weight.shape) == 3, "only support 3-dim batch matmul"

    # Topi batch matmul currently support NT mode. So, add transposes when it is not NT
    if transpose_a:
        data = _topi.transpose(data, (0, 2, 1))
    if not transpose_b:
        weight = _topi.transpose(weight, (0, 2, 1))
    return [_topi.nn.batch_matmul(data, weight)]


@register_compute("raf.op.tvm.batch_matmul")
def compute_batch_matmul_nn(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=False, transpose_b=False
    )


@register_compute("raf.op.tvm.batch_matmul_tn")
def compute_batch_matmul_tn(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=True, transpose_b=False
    )


@register_compute("raf.op.tvm.batch_matmul_tt")
def compute_batch_matmul_tt(attr, inputs, output_type):
    return compute_batch_matmul_general(
        attr, inputs, output_type, transpose_a=True, transpose_b=True
    )


_reg.register_injective_schedule("raf.op.tvm.batch_matmul")
_reg.register_injective_schedule("raf.op.tvm.batch_matmul_tn")
_reg.register_injective_schedule("raf.op.tvm.batch_matmul_tt")

_reg.register_strategy("raf.op.tvm.batch_matmul_nt", strategy.batch_matmul_strategy)


@register_compute("raf.op.tvm.softmax", level=15)
def compute_softmax(attr, inputs, output_type):
    return [_topi.nn.softmax(inputs[0])]


@generic_func
def schedule_softmax(attrs, outs, target):
    # FIXME: softmax is not an injective op so we should not use inject schedule.
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_softmax.register(["cuda", "gpu"])
def schedule_softmax_cuda(attrs, outs, _):
    out = outs[0]
    ndim = len(out.shape)
    sch = _tvm.te.create_schedule([out.op])
    num_thread = 64

    axis = attrs.axis
    axis = int(axis) if axis is not None else ndim - 1
    if axis >= ndim:
        axis %= ndim
    while axis < 0:
        axis += ndim

    if axis == 0:
        raise ValueError(
            "Internal error: softmax schedule does not support axis=0. "
            "This op should be dispatched to CuDNN dialect"
        )

    visited = set([out.op])

    def schedule(curr):
        if isinstance(curr.op, _tvm.te.ComputeOp) and curr.op not in visited:
            visited.add(curr.op)
            if isinstance(curr.op.body[0], _tvm.tir.expr.Reduce) and curr != out:
                _, inner = sch[curr].split(curr.op.reduce_axis[0], factor=num_thread)
                sch[curr].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
                sch[curr].compute_at(sch[out], out.op.axis[axis - 1])
                sch[curr].pragma(curr.op.axis[0], "auto_unroll_max_step", num_thread)
                sch[curr].pragma(curr.op.axis[0], "unroll_explicit", True)
            else:
                sch[curr].compute_inline()
        for inp in curr.op.input_tensors:
            schedule(inp)

    schedule(out)

    _, inner = sch[out].split(out.op.axis[axis], factor=num_thread)
    sch[out].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
    fused = sch[out].fuse(*out.op.axis[:axis])
    sch[out].bind(fused, _tvm.te.thread_axis("blockIdx.x"))
    return sch


_reg.register_schedule("raf.op.tvm.softmax", schedule_softmax)


@register_compute("raf.op.tvm.softmax_dx")
def compute_softmax_dx(attr, inputs, output_type):
    y, dy = inputs[0], inputs[1]
    axis = attr.axis
    return [(dy - _topi.sum(dy * y, axis, True)) * y]


@generic_func
def schedule_softmax_dx(attrs, outs, target):
    # FIXME: softmax_dx is not an injective op so we should not use inject schedule.
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_softmax_dx.register(["cuda", "gpu"])
def schedule_softmax_dx_cuda(attrs, outs, _):
    out = outs[0]
    ndim = len(out.shape)
    sch = _tvm.te.create_schedule([out.op])
    num_thread = 64

    axis = attrs.axis
    axis = int(axis) if axis is not None else ndim - 1
    if axis >= ndim:
        axis %= ndim
    while axis < 0:
        axis += ndim

    visited = set([out.op])

    def schedule(curr):
        if isinstance(curr.op, _tvm.te.ComputeOp) and curr.op not in visited:
            visited.add(curr.op)
            if isinstance(curr.op.body[0], _tvm.tir.expr.Reduce) and curr != out:
                if ndim > 1 and axis == ndim - 1:
                    # Cross-thread reduction schedule is applicable only for the last axis.
                    _, inner = sch[curr].split(curr.op.reduce_axis[0], factor=num_thread)
                    sch[curr].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
                    sch[curr].compute_at(sch[out], out.op.axis[axis - 1])
                    sch[curr].pragma(curr.op.axis[0], "auto_unroll_max_step", num_thread)
                    sch[curr].pragma(curr.op.axis[0], "unroll_explicit", True)
                else:
                    # General schedule for rest cases.
                    fused = sch[curr].fuse(*curr.op.axis)
                    outer, inner = sch[curr].split(fused, factor=num_thread)
                    sch[curr].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
                    sch[curr].bind(outer, _tvm.te.thread_axis("blockIdx.x"))
                    sch[curr].pragma(outer, "auto_unroll_max_step", num_thread)
                    sch[curr].pragma(outer, "unroll_explicit", True)
            else:
                sch[curr].compute_inline()
        for inp in curr.op.input_tensors:
            schedule(inp)

    schedule(out)

    if ndim > 1 and axis == ndim - 1:
        # Cross-thread reduction schedule is applicable only for the last axis.
        _, inner = sch[out].split(out.op.axis[axis], factor=num_thread)
        sch[out].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
        fused = sch[out].fuse(*out.op.axis[:axis])
        sch[out].bind(fused, _tvm.te.thread_axis("blockIdx.x"))
    else:
        # General schedule for rest cases.
        fused = sch[out].fuse(*out.op.axis)
        outer, inner = sch[out].split(fused, factor=num_thread)
        sch[out].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
        sch[out].bind(outer, _tvm.te.thread_axis("blockIdx.x"))

    return sch


_reg.register_schedule("raf.op.tvm.softmax_dx", schedule_softmax_dx)

_reg.register_schedule("raf.op.tvm.avg_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

_reg.register_schedule("raf.op.tvm.avg_pool2d_dx", strategy.schedule_pool_grad)

_reg.register_schedule("raf.op.tvm.max_pool2d", strategy.schedule_pool)
# TODO(@XIAO-XIA): cuda schedule should be complemented after the implementation of auto schedule

_reg.register_schedule("raf.op.tvm.max_pool2d_dx", strategy.schedule_pool_grad)

_reg.register_schedule("raf.op.tvm.adaptive_avg_pool2d", strategy.schedule_adaptive_pool)
_reg.register_schedule("raf.op.tvm.adaptive_avg_pool2d_dx", strategy.schedule_pool_grad)
_reg.register_schedule("raf.op.tvm.adaptive_max_pool2d", strategy.schedule_adaptive_pool)
_reg.register_schedule("raf.op.tvm.adaptive_max_pool2d_dx", strategy.schedule_pool_grad)


@generic_func
def schedule_log_softmax(attrs, outs, target):
    # Use the TVM schedules for other targets.
    with target:
        return _topi.generic.schedule_softmax(outs)


@schedule_log_softmax.register(["cuda", "gpu"])
def schedule_log_softmax_cuda(attrs, outs, _):
    """Override the CUDA schedule for better performance and fusion support."""
    out = outs[0]

    axis = attrs.axis
    ndim = len(out.shape)
    assert ndim == 2, "Only support 2-D log_softmax"
    axis = int(axis) if axis is not None else 1
    if axis >= ndim:
        axis %= ndim
    while axis < 0:
        axis += ndim

    sch = _tvm.te.create_schedule([out.op])
    thd_x = _tvm.te.thread_axis("threadIdx.x")
    blk_x = _tvm.te.thread_axis("blockIdx.x")

    # Schedule the final stage.
    (out_local,) = sch.cache_write([out], "local")
    sch[out_local].compute_inline()

    _, out_j_i = sch[out].split(out.op.axis[1], factor=32)
    sch[out].bind(out_j_i, thd_x)
    sch[out].bind(out.op.axis[0], blk_x)

    # Schedule the intermediate stages.
    visited = set([out.op])

    def schedule_dag(curr):
        if curr.op not in visited:
            visited.add(curr.op)
            if isinstance(curr.op, _tvm.te.tensor.ComputeOp):
                if isinstance(curr.op.body[0], _tvm.tir.expr.Reduce):
                    # Compute at reduce stage (e.g., max)
                    _, inner = sch[curr].split(curr.op.reduce_axis[0], factor=32)
                    sch[curr].bind(inner, thd_x)
                    sch[curr].compute_at(sch[out], out.op.axis[0])
                    sch[curr].pragma(curr.op.axis[0], "auto_unroll_max_step", 64)
                    sch[curr].pragma(curr.op.axis[0], "unroll_explicit", True)
                else:
                    # Inline elementwise stage (e.g., cast, exp).
                    sch[curr].compute_inline()

        for _inp in curr.op.input_tensors:
            schedule_dag(_inp)

    schedule_dag(out)
    return sch


_reg.register_schedule("raf.op.tvm.log_softmax", schedule_log_softmax)


@register_compute("raf.op.tvm.log_softmax_dx")
def compute_log_softmax_dx(attr, inputs, output_type):
    # The grad function of log_softmax decomposes log_softmax_dx to a series of RAF IR ops
    # so this function is not used. It only kept in case we want to have a powerful schedule
    # especially for this op in the future.
    y, dy = inputs[0], inputs[1]
    axis = attr.axis
    return [dy - _topi.exp(y) * _topi.sum(dy, axis, False)]


_reg.register_injective_schedule("raf.op.tvm.log_softmax_dx")


@register_compute("raf.op.tvm._contrib_dropout")
def compute_contrib_dropout(attr, inputs, output_type):
    # pylint: disable=import-outside-toplevel
    x = inputs[0]
    p = attr.rate
    if x.dtype != "float32" and x.dtype != "float64":
        raise TypeError(
            "input array of raf.dropout is expected to be the type of float32 "
            + "or float64, but received {}".format(x.dtype)
        )
    if p < 0.0 or p >= 1:
        raise ValueError("p is out of interval")
    retain_p = _tvm.tir.const(1 - p, x.dtype)
    mask = random.uniform(0, 1, x.shape)
    ret = _tvm.te.compute(
        x.shape,
        lambda *ix: _tvm.te.if_then_else(
            mask[ix] <= _tvm.tir.const(p, "float32"), _tvm.tir.const(0, x.dtype), x[ix] / retain_p
        ),
    )
    mask = _tvm.te.compute(
        x.shape,
        lambda *ix: _tvm.te.if_then_else(
            mask[ix] <= _tvm.tir.const(p, "float32"),
            _tvm.tir.const(0, "float32"),
            _tvm.tir.const(1 / (1 - p), "float32"),
        ),
    )
    # reserve_space is valid in cudnn only
    reserve_space_shape = ()
    if len(output_type.fields[-1].shape) > 0:
        # Reserve_space is not scalar type. It is dispatched from the base op
        from .._ffi.backend.cudnn import GetDropoutReserveSpaceSizeInBytes

        if GetDropoutReserveSpaceSizeInBytes:
            x_ty = _tvm.relay.TensorType(x.shape, dtype=x.dtype)
            reserve_space_shape = (GetDropoutReserveSpaceSizeInBytes(x_ty),)
    reserve_space = _topi.full(reserve_space_shape, dtype="uint8", fill_value=0.0)
    return [ret, mask, reserve_space]


_reg.register_injective_schedule("raf.op.tvm._contrib_dropout")


@register_compute("raf.op.tvm._contrib_dropout_dx")
def compute_contrib_dropout_dx(attr, inputs, output_type):
    dy = inputs[0]
    mask = inputs[1]
    assert _topi.utils.get_const_tuple(dy.shape) == _topi.utils.get_const_tuple(
        mask.shape
    ), "dy.shape %s != mask.shape %s" % (str(dy.shape), str(mask.shape))
    ret = _tvm.te.compute(dy.shape, lambda *idx: dy[idx] * _tvm.topi.cast(mask[idx], dy.dtype))
    return [ret]


_reg.register_injective_schedule("raf.op.tvm._contrib_dropout_dx")


@register_compute("raf.op.tvm.relu_dx")
@_tvm.te.tag_scope(tag=_tvm.topi.tag.ELEMWISE)
def compute_relu_dx(attr, inputs, output_type):
    grad_mode = attr.grad_mode
    if grad_mode == "both":
        data, dy = inputs[0], inputs[2]
    else:
        data, dy = inputs[0], inputs[1]
    # For y = relu(x), x or y can be used to calcluate graident
    # if both x and y are given, we use x here
    # Using x: return 0 if x < 0 else dy
    # Using y: return 0 if y == 0 else dy
    G = _tvm.te.compute(
        dy.shape,
        lambda *idx: _tvm.te.if_then_else(data[idx] <= 0, _tvm.tir.const(0, dy.dtype), dy[idx]),
    )
    return [G]


_reg.register_injective_schedule("raf.op.tvm.relu_dx")


@register_compute("raf.op.tvm.threshold")
def compute_threshold(attr, inputs, output_type):
    x = inputs[0]
    threshold = _tvm.tir.const(attr.threshold, x.dtype)
    value = _tvm.tir.const(attr.value, x.dtype)
    return [
        _tvm.te.compute(
            x.shape,
            lambda *idx: _tvm.te.if_then_else(x[idx] > threshold, x[idx], value),
            tag=_tvm.topi.tag.ELEMWISE,
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.threshold")


@register_compute("raf.op.tvm.threshold_dx")
def compute_threshold_dx(attr, inputs, output_type):
    x, dy = inputs[0], inputs[1]
    threshold = _tvm.tir.const(attr.threshold, x.dtype)
    return [
        _tvm.te.compute(
            dy.shape,
            lambda *idx: _tvm.te.if_then_else(
                x[idx] > threshold, dy[idx], _tvm.tir.const(0, dy.dtype)
            ),
            tag=_tvm.topi.tag.ELEMWISE,
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.threshold_dx")


@register_compute("raf.op.tvm.layer_norm_train")
def compute_layer_norm_train(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    x = inputs[0]

    # Determine the compute dtype. There are several cases:
    # 1) FP32 model: all inputs are FP32. dtyp3=FP32.
    # 2) FP16 model: all inputs are FP16. dtype=FP16.
    # 3) AMP model: x is FP16 but weight/bias are FP32. dtype=FP32.
    # In conclusion, the compute dtype should follow weight/bias dtype.
    # If weight/bias are None, then follow x dtype.
    orig_dtype = x.dtype
    dtype = x.dtype
    if attr.set_scale_bias:
        scale, bias = inputs[1], inputs[2]
        assert scale.dtype == bias.dtype
        dtype = scale.dtype
    else:
        scale, bias = None, None

    if x.dtype != dtype:
        x = _topi.cast(x, dtype)

    eps = _tvm.tir.const(attr.epsilon, dtype=dtype)
    axis = _topi.utils.get_const_int(attr.axis)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    def pad(data, target):
        newaxis = []
        for i in range(ndim):
            if i != axis:
                newaxis.append(i)
        return _topi.expand_like(data, target, newaxis)

    count = _tvm.tir.const(1, dtype=dtype)
    count *= x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    out = _topi.divide(_topi.subtract(x, x_mean), denominator)

    if scale is not None:
        pscale = pad(scale, out)
        out = _topi.multiply(pscale, out)
        out = _topi.add(out, pad(bias, out))

    if out.dtype != orig_dtype:
        out = _topi.cast(out, orig_dtype)

    # Calculate the shape of mean and var, which dimensions are the same as x but
    # are required to be 1-D when being outputs.
    idiff = ndim - 1 if scale is None else ndim - len(scale.shape)
    mean_var_shape = 1
    for i in x.shape[:idiff]:
        mean_var_shape *= i

    out_mean = _topi.reshape(x_mean, [mean_var_shape])
    out_var = _topi.reshape(x_var, [mean_var_shape])
    return [out, out_mean, out_var]


@register_compute("raf.op.tvm.layer_norm")
def compute_layer_norm(attr, inputs, output_type):
    outs = compute_layer_norm_train(attr, inputs, output_type)
    return [outs[0]]


@generic_func
def schedule_generic(attrs, outs, target):
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_generic.register(["cuda", "gpu"])
def schedule_generic_cuda(attrs, outs, target):
    with target:
        out = outs[0]
        s = cuda.injective.schedule_injective(outs)
        # fuse axes and split into bx and tx then bind
        scheduled_ops = []
        num_thread = 64

        def bind_axes(s, out):
            if (
                isinstance(out.op, _tvm.te.ComputeOp)
                and isinstance(out.op.body[0], _tvm.tir.expr.Reduce)
                and len(s[out].iter_var_attrs) == 0
                and out.op not in scheduled_ops
            ):
                scheduled_ops.append(out.op)
                fused = s[out].fuse(*s[out].op.axis)
                bx, tx = s[out].split(fused, factor=num_thread)
                s[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
                s[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
            for inp in out.op.input_tensors:
                bind_axes(s, inp)

        bind_axes(s, out)
        return s


@generic_func
def schedule_layer_norm(attrs, outs, target):
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_layer_norm.register(["cuda", "gpu"])
def schedule_layer_norm_cuda(attrs, outs, target):
    out = outs[0]
    num_thread = target.max_num_threads
    sch = _tvm.te.create_schedule([out.op])

    axis = _topi.utils.get_const_int(attrs.axis)
    axis = len(out.shape) + axis if axis < 0 else axis

    # Schedule ops except for the final one.
    visited = set([out.op])

    def schedule(curr):
        """A helper to traverse the compute DAG and 1) inline element-wise ops,
        2) fuse reduction ops.
        """
        if isinstance(curr.op, _tvm.te.ComputeOp) and curr.op not in visited:
            visited.add(curr.op)
            if isinstance(curr.op.body[0], _tvm.tir.expr.Reduce):
                _, inner = sch[curr].split(curr.op.reduce_axis[0], factor=num_thread)
                sch[curr].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
                if axis > 0:
                    sch[curr].compute_at(sch[out], out.op.axis[axis - 1])
                sch[curr].pragma(curr.op.axis[0], "auto_unroll_max_step", 32)
                sch[curr].pragma(curr.op.axis[0], "unroll_explicit", True)
            else:
                sch[curr].compute_inline()
        for inp in curr.op.input_tensors:
            schedule(inp)

    schedule(out)

    _, inner = sch[out].split(out.op.axis[axis], factor=num_thread)
    sch[out].bind(inner, _tvm.te.thread_axis("threadIdx.x"))
    fused = sch[out].fuse(*out.op.axis[:axis])
    sch[out].bind(fused, _tvm.te.thread_axis("blockIdx.x"))
    return sch


# Layer norm train is currently being offloaded to the CUDA kernel, so we don't craft
# an efficient schedule for it now.
_reg.register_schedule("raf.op.tvm.layer_norm_train", schedule_generic)

_reg.register_schedule("raf.op.tvm.layer_norm", schedule_layer_norm)


def compute_layer_norm_dx_common(attr, inputs, recompute_mean_var=True):
    # pylint: disable=too-many-locals
    set_scale = attr.set_scale_bias
    if set_scale:
        x, scale, dy = inputs[:3]
    else:
        scale = None
        x, dy = inputs[:2]

    orig_dtype = x.dtype
    dtype = x.dtype
    if scale is not None:
        dtype = scale.dtype

    # In the case of AMP model with FP32 scale, cast x and dy to FP32 to maintain the accuracy.
    if x.dtype != dtype:
        x = _topi.cast(x, dtype)
    if dy.dtype != dtype:
        dy = _topi.cast(dy, dtype)

    axis, eps = _topi.utils.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    count = x.shape[axis]
    reduce_axes = [axis]
    if recompute_mean_var:
        x_sum = _topi.sum(x, reduce_axes, keepdims=True)
        x_mean = _topi.divide(x_sum, count)

        sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
        sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
        x_var = _topi.divide(sq_diff_sum, count)
    else:
        # Calculate the shape of mean and var, which are 1-D from inputs but should match
        # the shape of x for broadcasting.
        idiff = ndim - 1 if scale is None else ndim - len(scale.shape)
        mean_var_shape = x.shape[:idiff] + [1 for _ in range(ndim - idiff)]
        in_mean, in_var = inputs[-2], inputs[-1]
        x_mean = _topi.reshape(in_mean, mean_var_shape)
        x_var = _topi.reshape(in_var, mean_var_shape)

    denominator = _topi.sqrt(_topi.add(x_var, eps))
    xmu = _topi.subtract(x, x_mean)

    bar_x = _topi.divide(xmu, denominator)
    w = _topi.divide(dy, denominator)

    def pad(data, target):
        newaxis = []
        for i in range(ndim):
            if i != axis:
                newaxis.append(i)
        return _topi.expand_like(data, target, newaxis)

    if set_scale:
        w = w * pad(scale, w)
    w_sum = _topi.sum(w, reduce_axes, keepdims=True)
    mean_w = _topi.divide(w_sum, count)
    w_times_bar_x = _topi.multiply(w, bar_x)
    w_times_bar_x_sum = _topi.sum(w_times_bar_x, reduce_axes, keepdims=True)
    mean_w_times_bar_x = _topi.divide(w_times_bar_x_sum, count)
    dx = _topi.subtract(w, mean_w)
    dx = _topi.subtract(dx, _topi.multiply(bar_x, mean_w_times_bar_x))
    if dx.dtype != orig_dtype:
        dx = _topi.cast(dx, orig_dtype)

    if set_scale:
        reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
        dw = _topi.sum(dy * (x - x_mean) / denominator, axis=reduce_axes)
        db = _topi.sum(dy, axis=reduce_axes)
        return [dx, dw, db]
    return [dx]


@register_compute("raf.op.tvm.layer_norm_dx")
def compute_layer_norm_dx(attr, inputs, output_type):
    return compute_layer_norm_dx_common(attr, inputs, recompute_mean_var=True)


_reg.register_schedule("raf.op.tvm.layer_norm_dx", schedule_generic)


@register_compute("raf.op.tvm.layer_norm_train_dx")
def compute_layer_norm_train_dx(attr, inputs, output_type):
    return compute_layer_norm_dx_common(attr, inputs, recompute_mean_var=False)


_reg.register_schedule("raf.op.tvm.layer_norm_train_dx", schedule_generic)

_reg.register_strategy("raf.op.tvm.conv2d", strategy.conv2d_strategy)

_reg.register_strategy("raf.op.tvm.conv2d_transpose", strategy.conv2d_transpose_strategy)


def declaration_conv2d_transpose_impl(data, kernel, strides, padding, out_dtype, output_padding):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = _topi.nn.conv2d_transpose_nchw_preprocess(
        data, kernel, strides, padding, out_dtype, (0, 0)
    )
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_c = _topi.nn.simplify(out_c)
    out_h = _topi.nn.simplify(in_h - filter_h + 1 + output_padding[0])
    out_w = _topi.nn.simplify(in_w - filter_w + 1 + output_padding[1])
    dc = _tvm.te.reduce_axis((0, in_c), name="dc")
    dh = _tvm.te.reduce_axis((0, filter_h), name="dh")
    dw = _tvm.te.reduce_axis((0, filter_w), name="dw")
    Output = _tvm.te.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: _tvm.tir.sum(
            data_pad[b, dc, h + dh, w + dw].astype(out_dtype)
            * kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",
    )
    return Output


@register_compute("raf.op.tvm.conv2d_dx")
def compute_conv2d_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"
    use_output = attr.use_output
    if use_output:
        W, dy = inputs[0], inputs[2]
    else:
        W, dy = inputs[0], inputs[1]
    X = _tvm.te.placeholder(shape=attr.kernel_size, dtype=dy.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_dx", schedule_generic)


@register_compute("raf.op.tvm.conv2d_dw")
def compute_conv2d_dw(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"

    use_output = attr.use_output
    if use_output:
        X, dy = inputs[0], inputs[2]
    else:
        X, dy = inputs[0], inputs[1]

    W = _tvm.te.placeholder(shape=attr.kernel_size, dtype=X.dtype)
    R = _topi.nn.conv2d(X, W, strides, padding, dilation, layout)
    grads = _tvm.te.gradient(R, [W], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_dw", schedule_generic)


@register_compute("raf.op.tvm.conv2d_transpose_dx")
def compute_conv2d_transpose_dx(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, output_padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.output_padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"
    use_output = attr.use_output
    if use_output:
        W, dy = inputs[0], inputs[2]
    else:
        W, dy = inputs[0], inputs[1]
    assert (
        W.shape[3] > 1 and W.shape[2] > 1
    ), "not support kernel size 1 for now. \
                                                See apache/tvm#8087"
    X = _tvm.te.placeholder(shape=attr.kernel_size, dtype=dy.dtype)
    R = _topi.x86.conv2d_transpose_nchw(X, W, strides, padding, dy.dtype, output_padding)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_transpose_dx", schedule_generic)


@register_compute("raf.op.tvm.conv2d_transpose_dw")
def compute_conv2d_transpose_dw(attr, inputs, output_type):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    strides, padding, output_padding, dilation, layout = (
        _topi.utils.get_const_tuple(attr.strides),
        _topi.utils.get_const_tuple(attr.padding),
        _topi.utils.get_const_tuple(attr.output_padding),
        _topi.utils.get_const_tuple(attr.dilation),
        attr.data_layout,
    )
    assert layout == "NCHW", "only support NCHW layout"
    assert dilation == (1, 1), "dilation is not supported yet"
    assert attr.groups == 1, "only support groups = 1"

    use_output = attr.use_output
    if use_output:
        X, dy = inputs[0], inputs[2]
    else:
        X, dy = inputs[0], inputs[1]

    W = _tvm.te.placeholder(shape=attr.kernel_size, dtype=X.dtype)
    R = _topi.x86.conv2d_transpose_nchw(X, W, strides, padding, dy.dtype, output_padding)

    grads = _tvm.te.gradient(R, [W], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.conv2d_transpose_dw", schedule_generic)


def average(data, axis):
    shape = _topi.utils.get_const_tuple(data.shape)
    shape = [shape[i] for i in axis]
    size = reduce(operator.mul, shape, 1)
    tot = _topi.sum(data, axis=axis)
    return _topi.divide(tot, size)


@register_compute("raf.op.tvm.batch_norm_train")
def batch_norm_train_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    x, running_m0, running_v0, w, b = inputs
    momentum, eps = attrs.momentum, attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1
    reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
    reduce_shape = [shape[i] for i in reduce_axes]
    reduce_size = reduce(operator.mul, reduce_shape, 1)

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    mean = average(x, axis=reduce_axes)
    x_sq = _topi.multiply(x, x)
    sq_mean = average(x_sq, axis=reduce_axes)
    mean_sq = _topi.multiply(mean, mean)
    var = sq_mean - mean_sq
    running_m = running_m0 * (1 - momentum) + mean * momentum
    running_v = running_v0 * (1 - momentum) + var * reduce_size / (reduce_size - 1) * momentum
    var_add_eps = _topi.add(var, eps)
    sqrt_var = _topi.sqrt(var_add_eps)
    scale = _topi.divide(w, sqrt_var)
    neg_mean = _topi.negative(mean)
    shift = _topi.multiply(neg_mean, scale)
    shift = _topi.add(shift, b)
    y = _topi.add(_topi.multiply(x, pad(scale)), pad(shift))
    return [y, running_m, running_v]


_reg.register_reduce_schedule("raf.op.tvm.batch_norm_train")


@register_compute("raf.op.tvm.batch_norm_infer")
def batch_norm_infer_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    x, running_m, running_v, w, b = inputs
    eps = attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    var_add_eps = _topi.add(running_v, eps)
    sqrt_var = _topi.sqrt(var_add_eps)
    scale = _topi.divide(w, sqrt_var)
    neg_mean = _topi.negative(running_m)
    shift = _topi.multiply(neg_mean, scale)
    shift = _topi.add(shift, b)
    y = _topi.add(_topi.multiply(x, pad(scale)), pad(shift))
    return [y]


_reg.register_injective_schedule("raf.op.tvm.batch_norm_infer")


@register_compute("raf.op.tvm.batch_norm_train_dxwb")
def batch_norm_train_dxwb_compute(attrs, inputs, output_type):  # pylint: disable=too-many-locals
    dy, x, w, _ = inputs
    eps = attrs.eps
    shape = _topi.utils.get_const_tuple(x.shape)
    ndim = len(shape)
    axis = 1
    num_newaxis = ndim - axis - 1
    reduce_axes = list(range(axis)) + list(range(axis + 1, ndim))
    reduce_shape = [shape[i] for i in reduce_axes]
    reduce_size = reduce(operator.mul, reduce_shape, 1)

    def pad(data):
        return _topi.expand_dims(data, axis=1, num_newaxis=num_newaxis)

    mean = average(x, axis=reduce_axes)
    x_sq = _topi.multiply(x, x)
    sq_mean = average(x_sq, axis=reduce_axes)
    mean_sq = _topi.multiply(mean, mean)
    var = sq_mean - mean_sq
    inv_sqrt_var = 1 / _topi.sqrt(var + eps)
    sum_dy_x = _topi.sum(dy * x, axis=reduce_axes)
    sum_dy = _topi.sum(dy, axis=reduce_axes)
    db = sum_dy
    dw = (sum_dy_x - mean * sum_dy) * inv_sqrt_var
    dx = (
        dy - pad(db / reduce_size) - (x - pad(mean)) * pad(dw * inv_sqrt_var) / reduce_size
    ) * pad(w * inv_sqrt_var)
    return [dx, dw, db]


_reg.register_reduce_schedule("raf.op.tvm.batch_norm_train_dxwb")
