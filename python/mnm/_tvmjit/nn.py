# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
from .._lib import register_compute
from .._lib import generic_func
from .._lib import topi as _topi
from .._lib import tvm as _tvm
from .._lib import _reg
from .._lib import strategy

@register_compute("mnm.op.take_dx")
def take_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    x, y, dy, indices = inputs
    axis = int(attrs.axis)
    idim = len(indices.shape)
    # infer axis when negative
    dim = len(x.shape)
    if -dim < axis < 0:
        axis = dim + axis
    shape = dy.shape[:axis + idim] + [x.shape[axis],] + dy.shape[axis + idim:]
    A = _tvm.te.compute(shape, lambda *idx:
                        _tvm.tir.if_then_else(idx[axis + idim] == indices[idx[axis: axis + idim]],
                                              dy[idx[:axis + idim] + idx[axis + idim + 1:]],
                                              _tvm.tir.const(0, dy.dtype)))
    B = _topi.sum(A, axis=tuple(range(axis, axis + idim)))
    return [B]


_reg.register_injective_schedule("mnm.op.take_dx")

_reg.register_strategy("mnm.op.dense", strategy.dense_strategy)

_reg.register_strategy("mnm.op.batch_matmul", strategy.batch_matmul_strategy)

_reg.register_strategy("mnm.op.softmax", strategy.softmax_strategy)

@register_compute("mnm.op.softmax_dx")
def compute_softmax_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs[0], inputs[1], inputs[2]
    axis = attr.axis
    dy_sum = _topi.sum(dy * y, axis=axis, keepdims=True)
    dy_sum = _topi.repeat(dy_sum, repeats=int(x.shape[axis]), axis=axis)
    return [y * (dy - dy_sum)]

_reg.register_injective_schedule("mnm.op.softmax_dx")

@register_compute("mnm.op.relu_dx")
def compute_relu_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    X, dy, y = inputs[0], inputs[1], inputs[2]
    R = _topi.nn.relu(X)
    grads = _tvm.te.gradient(R, [X], head=dy)
    return grads

_reg.register_injective_schedule("mnm.op.relu_dx")

@register_compute("mnm.op.layer_norm")
def compute_layer_norm(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=too-many-locals
    x = inputs[0]
    axis, eps = _topi.util.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis
    count = _tvm.tir.const(1, dtype=x.dtype)
    count *= x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    out = _topi.divide(_topi.subtract(x, x_mean), denominator)
    return [out]

@generic_func
def schedule_layer_norm(attrs, outs, target):
    # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_injective(outs)

@schedule_layer_norm.register(["cuda", "gpu"])
def schedule_layer_norm_cuda(attrs, outs, target):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    with target:
        out = outs[0]
        s = _topi.cuda.schedule_injective(outs)
        # fuse axes and split into bx and tx then bind
        scheduled_ops = []
        num_thread = 64
        def bind_axes(s, out):
            if isinstance(out.op, _tvm.te.ComputeOp) and out.op.tag == 'comm_reduce' \
                and out.op not in scheduled_ops:
                scheduled_ops.append(out.op)
                fused = s[out].fuse(*s[out].op.axis)
                bx, tx = s[out].split(fused, factor=num_thread)
                s[out].bind(bx, _tvm.te.thread_axis("blockIdx.x"))
                s[out].bind(tx, _tvm.te.thread_axis("threadIdx.x"))
            for inp in out.op.input_tensors:
                bind_axes(s, inp)
        bind_axes(s, out)
        return s

_reg.register_schedule("mnm.op.layer_norm", schedule_layer_norm)

@register_compute("mnm.op.layer_norm_dx")
def compute_layer_norm_dx(attr, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=too-many-locals
    # pylint: disable=unused-variable
    x, y, dy = inputs
    axis, eps = _topi.util.get_const_int(attr.axis), _tvm.tir.const(attr.epsilon, dtype=x.dtype)
    ndim = len(x.shape)
    if axis < 0:
        axis = ndim + axis

    count = x.shape[axis]
    reduce_axes = [axis]
    x_sum = _topi.sum(x, reduce_axes, keepdims=True)
    x_mean = _topi.divide(x_sum, count)
    sq_diff = _topi.power(_topi.subtract(x, x_mean), 2)
    sq_diff_sum = _topi.sum(sq_diff, reduce_axes, keepdims=True)
    x_var = _topi.divide(sq_diff_sum, count)
    denominator = _topi.sqrt(_topi.add(x_var, eps))
    xmu = _topi.subtract(x, x_mean)

    bar_x = _topi.divide(xmu, denominator)
    w = _topi.divide(dy, denominator)
    w_sum = _topi.sum(w, reduce_axes, keepdims=True)
    mean_w = _topi.divide(w_sum, count)
    w_times_bar_x = _topi.multiply(w, bar_x)
    w_times_bar_x_sum = _topi.sum(w_times_bar_x, reduce_axes, keepdims=True)
    mean_w_times_bar_x = _topi.divide(w_times_bar_x_sum, count)
    dx = _topi.subtract(w, mean_w)
    dx = _topi.subtract(dx, _topi.multiply(bar_x, mean_w_times_bar_x))
    return [dx]

_reg.register_schedule("mnm.op.layer_norm_dx", schedule_layer_norm)
