# pylint: disable=invalid-name,line-too-long,too-many-lines
# pylint: disable=too-many-arguments,redefined-builtin,redefined-outer-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Auto generated. Do not touch."""
from mnm._ffi.op import GetOp
from mnm._lib import relay
from . import op_utils

__all__ = [
    "_allgather", "_allreduce", "_broadcast", "_contrib_dropout", "_contrib_dropout_dx",
    "_recv", "_reduce", "_reduce_scatter", "_send", "abs",
    "adaptive_avg_pool2d", "adaptive_avg_pool2d_dx", "adaptive_max_pool2d", "adaptive_max_pool2d_dx", "add",
    "add_event", "adv_index", "adv_index_dx", "all", "any",
    "arange", "argmax", "argmin", "argsort", "argwhere",
    "atan", "avg_pool2d", "avg_pool2d_dx", "batch_flatten", "batch_matmul",
    "batch_matmul_nt", "batch_matmul_tn", "batch_matmul_tt", "batch_norm_infer", "batch_norm_train",
    "batch_norm_train_dxwb", "bias_add", "broadcast_to", "broadcast_to_like", "cast",
    "cast_like", "ceil", "clip", "clip_dx", "collapse_sum_like",
    "compiler_begin", "compiler_end", "concatenate", "concatenate_dx", "conv2d",
    "conv2d_dw", "conv2d_dx", "conv2d_transpose", "conv2d_transpose_dw", "conv2d_transpose_dx",
    "copy", "cos", "cross_entropy", "cross_entropy_dpred", "cross_entropy_dtrue",
    "cumsum", "dense", "device_copy", "divide", "embedding",
    "embedding_dx", "equal", "erf", "erf_dx", "exp",
    "expand_dims", "floor", "floor_divide", "full", "full_like",
    "gather", "gather_dx", "gather_nd", "gather_nd_dx", "gelu",
    "gelu_dx", "get_kept_dims", "get_reduce_axis", "get_valid_counts", "greater",
    "greater_equal", "layer_norm", "layer_norm_dx", "left_shift", "less",
    "less_equal", "log", "log2", "log_softmax", "log_softmax_dx",
    "logical_and", "logical_not", "matmul", "matmul_nt", "matmul_tn",
    "matmul_tt", "max", "max_pool2d", "max_pool2d_dx", "maximum",
    "mean", "mean_dx", "mesh_grid", "min", "minimum",
    "mod", "multiply", "ndarray_size", "negative", "nll_loss",
    "nll_loss_dpred", "nll_loss_dtrue", "non_max_suppression", "not_equal", "one_hot",
    "ones", "ones_like", "pad", "power", "prod",
    "prod_dx", "relu", "relu_dx", "repeat", "repeat_dx",
    "reshape", "resize2d", "resize2d_dx", "reverse", "reverse_sequence",
    "right_shift", "roi_align", "roi_align_dx", "round", "rsqrt",
    "scatter", "scatter_dx", "sequence_mask", "set_stream", "sgd",
    "shape", "sigmoid", "sigmoid_dx", "sign", "sin",
    "smooth_l1_loss", "smooth_l1_loss_dpred", "smooth_l1_loss_dtrue", "softmax", "softmax_dx",
    "sort", "split", "sqrt", "sqrt_dx", "squeeze",
    "stack", "stream_barrier", "stream_sync", "strided_slice", "strided_slice_dx",
    "subtract", "sum", "sum_dx", "swap_axis", "take",
    "take_dx", "tanh", "tanh_dx", "threefry_generate", "threefry_split",
    "threshold", "threshold_dx", "topk", "transpose", "transpose_dx",
    "trunc", "upper_bound_argwhere", "vm_alloc_storage", "vm_alloc_tensor", "vm_free",
    "vm_infer_type", "vm_invoke_op", "vm_set_shape", "wait_event", "where",
    "zeros", "zeros_like",
]

def _allgather(x, axis, attrs=None):
    op = GetOp("mnm.op._allgather")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def _allreduce(x, computation="sum", attrs=None):
    op = GetOp("mnm.op._allreduce")
    x = op_utils.to_tensor_tuple(x)
    computation = op_utils.to_string(computation)
    return relay.Call(op, [x, computation], attrs)

def _broadcast(x, root, attrs=None):
    op = GetOp("mnm.op._broadcast")
    x = op_utils.to_tensor_tuple(x)
    root = op_utils.to_int(root)
    return relay.Call(op, [x, root], attrs)

def _contrib_dropout(x, p=0.5, in_states=None, attrs=None):
    op = GetOp("mnm.op._contrib_dropout")
    x = op_utils.to_tensor(x)
    p = op_utils.to_double(p)
    in_states = op_utils.to_tensor(in_states)
    return relay.Call(op, [x, p, in_states], attrs)

def _contrib_dropout_dx(dy, mask, reserve_space, p=0.5, attrs=None):
    op = GetOp("mnm.op._contrib_dropout_dx")
    dy = op_utils.to_tensor(dy)
    mask = op_utils.to_tensor(mask)
    reserve_space = op_utils.to_tensor(reserve_space)
    p = op_utils.to_double(p)
    return relay.Call(op, [dy, mask, reserve_space, p], attrs)

def _recv(peer, shape, dtype="float32", token=None, attrs=None):
    op = GetOp("mnm.op._recv")
    peer = op_utils.to_int(peer)
    shape = op_utils.to_int_tuple(shape)
    dtype = op_utils.to_string(dtype)
    token = op_utils.to_tensor(token)
    return relay.Call(op, [peer, shape, dtype, token], attrs)

def _reduce(x, root, computation="sum", attrs=None):
    op = GetOp("mnm.op._reduce")
    x = op_utils.to_tensor_tuple(x)
    root = op_utils.to_int(root)
    computation = op_utils.to_string(computation)
    return relay.Call(op, [x, root, computation], attrs)

def _reduce_scatter(x, computation="sum", attrs=None):
    op = GetOp("mnm.op._reduce_scatter")
    x = op_utils.to_tensor_tuple(x)
    computation = op_utils.to_string(computation)
    return relay.Call(op, [x, computation], attrs)

def _send(x, peer, token=None, attrs=None):
    op = GetOp("mnm.op._send")
    x = op_utils.to_tensor(x)
    peer = op_utils.to_int(peer)
    token = op_utils.to_tensor(token)
    return relay.Call(op, [x, peer, token], attrs)

def abs(x, attrs=None):
    op = GetOp("mnm.op.abs")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def adaptive_avg_pool2d(x, shape, layout="NCHW", attrs=None):
    op = GetOp("mnm.op.adaptive_avg_pool2d")
    x = op_utils.to_tensor(x)
    shape = op_utils.to_int_tuple(shape)
    layout = op_utils.to_string(layout)
    return relay.Call(op, [x, shape, layout], attrs)

def adaptive_avg_pool2d_dx(x, y, dy, shape, attrs=None):
    op = GetOp("mnm.op.adaptive_avg_pool2d_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    return relay.Call(op, [x, y, dy, shape], attrs)

def adaptive_max_pool2d(x, shape, layout="NCHW", attrs=None):
    op = GetOp("mnm.op.adaptive_max_pool2d")
    x = op_utils.to_tensor(x)
    shape = op_utils.to_int_tuple(shape)
    layout = op_utils.to_string(layout)
    return relay.Call(op, [x, shape, layout], attrs)

def adaptive_max_pool2d_dx(x, y, dy, shape, attrs=None):
    op = GetOp("mnm.op.adaptive_max_pool2d_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    return relay.Call(op, [x, y, dy, shape], attrs)

def add(x1, x2, out=None, where=None, attrs=None):
    op = GetOp("mnm.op.add")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    out = op_utils.to_any(out)
    where = op_utils.to_any(where)
    return relay.Call(op, [x1, x2, out, where], attrs)

def add_event(event_id, stream_id=-1, attrs=None):
    op = GetOp("mnm.op.add_event")
    event_id = op_utils.to_int(event_id)
    stream_id = op_utils.to_int(stream_id)
    return relay.Call(op, [event_id, stream_id], attrs)

def adv_index(inputs, attrs=None):
    op = GetOp("mnm.op.adv_index")
    inputs = op_utils.to_tensor_tuple(inputs)
    return relay.Call(op, [inputs], attrs)

def adv_index_dx(dy, inputs, attrs=None):
    op = GetOp("mnm.op.adv_index_dx")
    dy = op_utils.to_tensor(dy)
    inputs = op_utils.to_tensor_tuple(inputs)
    return relay.Call(op, [dy, inputs], attrs)

def all(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.all")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def any(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.any")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def arange(start, stop, step, dtype="float32", device="cpu", attrs=None):
    op = GetOp("mnm.op.arange")
    start = op_utils.to_tensor(start)
    stop = op_utils.to_tensor(stop)
    step = op_utils.to_tensor(step)
    dtype = op_utils.to_string(dtype)
    device = op_utils.to_string(device)
    return relay.Call(op, [start, stop, step, dtype, device], attrs)

def argmax(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.argmax")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def argmin(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.argmin")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def argsort(data, axis=-1, is_ascend=True, dtype="int32", attrs=None):
    op = GetOp("mnm.op.argsort")
    data = op_utils.to_tensor(data)
    axis = op_utils.to_int(axis)
    is_ascend = op_utils.to_bool(is_ascend)
    dtype = op_utils.to_string(dtype)
    return relay.Call(op, [data, axis, is_ascend, dtype], attrs)

def argwhere(condition, attrs=None):
    op = GetOp("mnm.op.argwhere")
    condition = op_utils.to_tensor(condition)
    return relay.Call(op, [condition], attrs)

def atan(x, attrs=None):
    op = GetOp("mnm.op.atan")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def avg_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True, layout="NCHW", attrs=None):
    op = GetOp("mnm.op.avg_pool2d")
    x = op_utils.to_tensor(x)
    kernel = op_utils.to_int_tuple(kernel)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    ceil_mode = op_utils.to_bool(ceil_mode)
    include_pad = op_utils.to_bool(include_pad)
    layout = op_utils.to_string(layout)
    return relay.Call(op, [x, kernel, stride, padding, dilation, ceil_mode, include_pad, layout], attrs)

def avg_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad, attrs=None):
    op = GetOp("mnm.op.avg_pool2d_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    kernel = op_utils.to_int_tuple(kernel)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    ceil_mode = op_utils.to_bool(ceil_mode)
    include_pad = op_utils.to_bool(include_pad)
    return relay.Call(op, [x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad], attrs)

def batch_flatten(x, attrs=None):
    op = GetOp("mnm.op.batch_flatten")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def batch_matmul(x1, x2, attrs=None):
    op = GetOp("mnm.op.batch_matmul")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def batch_matmul_nt(x1, x2, attrs=None):
    op = GetOp("mnm.op.batch_matmul_nt")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def batch_matmul_tn(x1, x2, attrs=None):
    op = GetOp("mnm.op.batch_matmul_tn")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def batch_matmul_tt(x1, x2, attrs=None):
    op = GetOp("mnm.op.batch_matmul_tt")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def batch_norm_infer(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05, attrs=None):
    op = GetOp("mnm.op.batch_norm_infer")
    x = op_utils.to_tensor(x)
    running_mean = op_utils.to_tensor(running_mean)
    running_var = op_utils.to_tensor(running_var)
    w = op_utils.to_tensor(w)
    b = op_utils.to_tensor(b)
    momentum = op_utils.to_double(momentum)
    eps = op_utils.to_double(eps)
    return relay.Call(op, [x, running_mean, running_var, w, b, momentum, eps], attrs)

def batch_norm_train(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05, attrs=None):
    op = GetOp("mnm.op.batch_norm_train")
    x = op_utils.to_tensor(x)
    running_mean = op_utils.to_tensor(running_mean)
    running_var = op_utils.to_tensor(running_var)
    w = op_utils.to_tensor(w)
    b = op_utils.to_tensor(b)
    momentum = op_utils.to_double(momentum)
    eps = op_utils.to_double(eps)
    return relay.Call(op, [x, running_mean, running_var, w, b, momentum, eps], attrs)

def batch_norm_train_dxwb(dy, x, w, b, eps, attrs=None):
    op = GetOp("mnm.op.batch_norm_train_dxwb")
    dy = op_utils.to_tensor(dy)
    x = op_utils.to_tensor(x)
    w = op_utils.to_tensor(w)
    b = op_utils.to_tensor(b)
    eps = op_utils.to_double(eps)
    return relay.Call(op, [dy, x, w, b, eps], attrs)

def bias_add(x, bias, axis=1, attrs=None):
    op = GetOp("mnm.op.bias_add")
    x = op_utils.to_tensor(x)
    bias = op_utils.to_tensor(bias)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, bias, axis], attrs)

def broadcast_to(x, shape, attrs=None):
    op = GetOp("mnm.op.broadcast_to")
    x = op_utils.to_tensor(x)
    shape = op_utils.to_int_tuple(shape)
    return relay.Call(op, [x, shape], attrs)

def broadcast_to_like(x, broadcast_type, attrs=None):
    op = GetOp("mnm.op.broadcast_to_like")
    x = op_utils.to_tensor(x)
    broadcast_type = op_utils.to_tensor(broadcast_type)
    return relay.Call(op, [x, broadcast_type], attrs)

def cast(data, dtype, attrs=None):
    op = GetOp("mnm.op.cast")
    data = op_utils.to_tensor(data)
    dtype = op_utils.to_string(dtype)
    return relay.Call(op, [data, dtype], attrs)

def cast_like(data, dtype_like, attrs=None):
    op = GetOp("mnm.op.cast_like")
    data = op_utils.to_tensor(data)
    dtype_like = op_utils.to_tensor(dtype_like)
    return relay.Call(op, [data, dtype_like], attrs)

def ceil(x, attrs=None):
    op = GetOp("mnm.op.ceil")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def clip(x, a_min, a_max, attrs=None):
    op = GetOp("mnm.op.clip")
    x = op_utils.to_tensor(x)
    a_min = op_utils.to_double(a_min)
    a_max = op_utils.to_double(a_max)
    return relay.Call(op, [x, a_min, a_max], attrs)

def clip_dx(x, dy, a_min, a_max, attrs=None):
    op = GetOp("mnm.op.clip_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    a_min = op_utils.to_double(a_min)
    a_max = op_utils.to_double(a_max)
    return relay.Call(op, [x, dy, a_min, a_max], attrs)

def collapse_sum_like(x, shape, attrs=None):
    op = GetOp("mnm.op.collapse_sum_like")
    x = op_utils.to_tensor(x)
    shape = op_utils.to_int_tuple(shape)
    return relay.Call(op, [x, shape], attrs)

def compiler_begin(x, attrs=None):
    op = GetOp("mnm.op.compiler_begin")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def compiler_end(x, attrs=None):
    op = GetOp("mnm.op.compiler_end")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def concatenate(x, axis=0, attrs=None):
    op = GetOp("mnm.op.concatenate")
    x = op_utils.to_tensor_tuple(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def concatenate_dx(x, axis=0, attrs=None):
    op = GetOp("mnm.op.concatenate_dx")
    x = op_utils.to_tensor_tuple(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def conv2d(x, w, stride=1, padding=0, dilation=1, groups=1, layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", attrs=None):
    op = GetOp("mnm.op.conv2d")
    x = op_utils.to_tensor(x)
    w = op_utils.to_tensor(w)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    layout = op_utils.to_string(layout)
    kernel_layout = op_utils.to_string(kernel_layout)
    out_layout = op_utils.to_string(out_layout)
    return relay.Call(op, [x, w, stride, padding, dilation, groups, layout, kernel_layout, out_layout], attrs)

def conv2d_dw(x_or_w, y, dy, shape, stride, padding, dilation, groups, attrs=None):
    op = GetOp("mnm.op.conv2d_dw")
    x_or_w = op_utils.to_tensor(x_or_w)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    return relay.Call(op, [x_or_w, y, dy, shape, stride, padding, dilation, groups], attrs)

def conv2d_dx(x_or_w, y, dy, shape, stride, padding, dilation, groups, attrs=None):
    op = GetOp("mnm.op.conv2d_dx")
    x_or_w = op_utils.to_tensor(x_or_w)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    return relay.Call(op, [x_or_w, y, dy, shape, stride, padding, dilation, groups], attrs)

def conv2d_transpose(x, w, stride=1, padding=0, output_padding=0, dilation=1, groups=1, layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", attrs=None):
    op = GetOp("mnm.op.conv2d_transpose")
    x = op_utils.to_tensor(x)
    w = op_utils.to_tensor(w)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    output_padding = op_utils.to_int_tuple(output_padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    layout = op_utils.to_string(layout)
    kernel_layout = op_utils.to_string(kernel_layout)
    out_layout = op_utils.to_string(out_layout)
    return relay.Call(op, [x, w, stride, padding, output_padding, dilation, groups, layout, kernel_layout, out_layout], attrs)

def conv2d_transpose_dw(x_or_w, y, dy, shape, stride, padding, output_padding, dilation, groups, attrs=None):
    op = GetOp("mnm.op.conv2d_transpose_dw")
    x_or_w = op_utils.to_tensor(x_or_w)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    output_padding = op_utils.to_int_tuple(output_padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    return relay.Call(op, [x_or_w, y, dy, shape, stride, padding, output_padding, dilation, groups], attrs)

def conv2d_transpose_dx(x_or_w, y, dy, shape, stride, padding, output_padding, dilation, groups, attrs=None):
    op = GetOp("mnm.op.conv2d_transpose_dx")
    x_or_w = op_utils.to_tensor(x_or_w)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    shape = op_utils.to_int_tuple(shape)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    output_padding = op_utils.to_int_tuple(output_padding)
    dilation = op_utils.to_int_tuple(dilation)
    groups = op_utils.to_int(groups)
    return relay.Call(op, [x_or_w, y, dy, shape, stride, padding, output_padding, dilation, groups], attrs)

def copy(x, attrs=None):
    op = GetOp("mnm.op.copy")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def cos(x, attrs=None):
    op = GetOp("mnm.op.cos")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def cross_entropy(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.cross_entropy")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def cross_entropy_dpred(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.cross_entropy_dpred")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def cross_entropy_dtrue(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.cross_entropy_dtrue")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def cumsum(x, axis, dtype="float32", exclusive=False, attrs=None):
    op = GetOp("mnm.op.cumsum")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    dtype = op_utils.to_string(dtype)
    exclusive = op_utils.to_bool(exclusive)
    return relay.Call(op, [x, axis, dtype, exclusive], attrs)

def dense(x1, x2, attrs=None):
    op = GetOp("mnm.op.dense")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def device_copy(data, src_dev_type=0, dst_dev_type=0, attrs=None):
    op = GetOp("mnm.op.device_copy")
    data = op_utils.to_tensor(data)
    src_dev_type = op_utils.to_int(src_dev_type)
    dst_dev_type = op_utils.to_int(dst_dev_type)
    return relay.Call(op, [data, src_dev_type, dst_dev_type], attrs)

def divide(x1, x2, attrs=None):
    op = GetOp("mnm.op.divide")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def embedding(x, indices, attrs=None):
    op = GetOp("mnm.op.embedding")
    x = op_utils.to_tensor(x)
    indices = op_utils.to_tensor(indices)
    return relay.Call(op, [x, indices], attrs)

def embedding_dx(dy, indices, num_weight, attrs=None):
    op = GetOp("mnm.op.embedding_dx")
    dy = op_utils.to_tensor(dy)
    indices = op_utils.to_tensor(indices)
    num_weight = op_utils.to_int_tuple(num_weight)
    return relay.Call(op, [dy, indices, num_weight], attrs)

def equal(x1, x2, attrs=None):
    op = GetOp("mnm.op.equal")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def erf(x, attrs=None):
    op = GetOp("mnm.op.erf")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def erf_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.erf_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def exp(x, attrs=None):
    op = GetOp("mnm.op.exp")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def expand_dims(x, axis, num_newaxis=1, attrs=None):
    op = GetOp("mnm.op.expand_dims")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    num_newaxis = op_utils.to_int(num_newaxis)
    return relay.Call(op, [x, axis, num_newaxis], attrs)

def floor(x, attrs=None):
    op = GetOp("mnm.op.floor")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def floor_divide(x1, x2, attrs=None):
    op = GetOp("mnm.op.floor_divide")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def full(fill_value, shape, dtype="int32", device="cpu", attrs=None):
    op = GetOp("mnm.op.full")
    fill_value = op_utils.to_double(fill_value)
    shape = op_utils.to_int_tuple(shape)
    dtype = op_utils.to_string(dtype)
    device = op_utils.to_string(device)
    return relay.Call(op, [fill_value, shape, dtype, device], attrs)

def full_like(data, fill_value, attrs=None):
    op = GetOp("mnm.op.full_like")
    data = op_utils.to_tensor(data)
    fill_value = op_utils.to_double(fill_value)
    return relay.Call(op, [data, fill_value], attrs)

def gather(data, axis, indices, attrs=None):
    op = GetOp("mnm.op.gather")
    data = op_utils.to_tensor(data)
    axis = op_utils.to_int(axis)
    indices = op_utils.to_tensor(indices)
    return relay.Call(op, [data, axis, indices], attrs)

def gather_dx(data, axis, indices, dy, attrs=None):
    op = GetOp("mnm.op.gather_dx")
    data = op_utils.to_tensor(data)
    axis = op_utils.to_int(axis)
    indices = op_utils.to_tensor(indices)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [data, axis, indices, dy], attrs)

def gather_nd(data, indices, attrs=None):
    op = GetOp("mnm.op.gather_nd")
    data = op_utils.to_tensor(data)
    indices = op_utils.to_tensor(indices)
    return relay.Call(op, [data, indices], attrs)

def gather_nd_dx(data, indices, dy, attrs=None):
    op = GetOp("mnm.op.gather_nd_dx")
    data = op_utils.to_tensor(data)
    indices = op_utils.to_tensor(indices)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [data, indices, dy], attrs)

def gelu(x, attrs=None):
    op = GetOp("mnm.op.gelu")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def gelu_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.gelu_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def get_kept_dims(x1, x2, attrs=None):
    op = GetOp("mnm.op.get_kept_dims")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def get_reduce_axis(x1, x2, attrs=None):
    op = GetOp("mnm.op.get_reduce_axis")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def get_valid_counts(data, score_threshold, id_index=0, score_index=1, attrs=None):
    op = GetOp("mnm.op.get_valid_counts")
    data = op_utils.to_tensor(data)
    score_threshold = op_utils.to_tensor(score_threshold)
    id_index = op_utils.to_int(id_index)
    score_index = op_utils.to_int(score_index)
    return relay.Call(op, [data, score_threshold, id_index, score_index], attrs)

def greater(x1, x2, attrs=None):
    op = GetOp("mnm.op.greater")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def greater_equal(x1, x2, attrs=None):
    op = GetOp("mnm.op.greater_equal")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def layer_norm(x, scale=None, bias=None, axis=-1, eps=1e-05, attrs=None):
    op = GetOp("mnm.op.layer_norm")
    x = op_utils.to_tensor(x)
    scale = op_utils.to_tensor(scale)
    bias = op_utils.to_tensor(bias)
    axis = op_utils.to_int(axis)
    eps = op_utils.to_double(eps)
    return relay.Call(op, [x, scale, bias, axis, eps], attrs)

def layer_norm_dx(x, scale, dy, axis=-1, eps=1e-05, attrs=None):
    op = GetOp("mnm.op.layer_norm_dx")
    x = op_utils.to_tensor(x)
    scale = op_utils.to_tensor(scale)
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int(axis)
    eps = op_utils.to_double(eps)
    return relay.Call(op, [x, scale, dy, axis, eps], attrs)

def left_shift(x1, x2, attrs=None):
    op = GetOp("mnm.op.left_shift")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def less(x1, x2, attrs=None):
    op = GetOp("mnm.op.less")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def less_equal(x1, x2, attrs=None):
    op = GetOp("mnm.op.less_equal")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def log(x, attrs=None):
    op = GetOp("mnm.op.log")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def log2(x, attrs=None):
    op = GetOp("mnm.op.log2")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def log_softmax(x, axis=-1, attrs=None):
    op = GetOp("mnm.op.log_softmax")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def log_softmax_dx(x, y, dy, axis=-1, attrs=None):
    op = GetOp("mnm.op.log_softmax_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, y, dy, axis], attrs)

def logical_and(x1, x2, attrs=None):
    op = GetOp("mnm.op.logical_and")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def logical_not(x, attrs=None):
    op = GetOp("mnm.op.logical_not")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def matmul(x1, x2, attrs=None):
    op = GetOp("mnm.op.matmul")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def matmul_nt(x1, x2, attrs=None):
    op = GetOp("mnm.op.matmul_nt")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def matmul_tn(x1, x2, attrs=None):
    op = GetOp("mnm.op.matmul_tn")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def matmul_tt(x1, x2, attrs=None):
    op = GetOp("mnm.op.matmul_tt")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def max(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.max")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def max_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True, layout="NCHW", attrs=None):
    op = GetOp("mnm.op.max_pool2d")
    x = op_utils.to_tensor(x)
    kernel = op_utils.to_int_tuple(kernel)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    ceil_mode = op_utils.to_bool(ceil_mode)
    include_pad = op_utils.to_bool(include_pad)
    layout = op_utils.to_string(layout)
    return relay.Call(op, [x, kernel, stride, padding, dilation, ceil_mode, include_pad, layout], attrs)

def max_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad, attrs=None):
    op = GetOp("mnm.op.max_pool2d_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    kernel = op_utils.to_int_tuple(kernel)
    stride = op_utils.to_int_tuple(stride)
    padding = op_utils.to_int_tuple(padding)
    dilation = op_utils.to_int_tuple(dilation)
    ceil_mode = op_utils.to_bool(ceil_mode)
    include_pad = op_utils.to_bool(include_pad)
    return relay.Call(op, [x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad], attrs)

def maximum(x1, x2, attrs=None):
    op = GetOp("mnm.op.maximum")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def mean(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.mean")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def mean_dx(dy, axis=(), x_shape=None, keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.mean_dx")
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int_tuple(axis)
    x_shape = op_utils.to_int_tuple(x_shape)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [dy, axis, x_shape, keepdims, exclude], attrs)

def mesh_grid(x, attrs=None):
    op = GetOp("mnm.op.mesh_grid")
    x = op_utils.to_tensor_tuple(x)
    return relay.Call(op, [x], attrs)

def min(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.min")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def minimum(x1, x2, attrs=None):
    op = GetOp("mnm.op.minimum")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def mod(x1, x2, attrs=None):
    op = GetOp("mnm.op.mod")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def multiply(x1, x2, attrs=None):
    op = GetOp("mnm.op.multiply")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def ndarray_size(x, attrs=None):
    op = GetOp("mnm.op.ndarray_size")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def negative(x, attrs=None):
    op = GetOp("mnm.op.negative")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def nll_loss(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.nll_loss")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def nll_loss_dpred(dy, y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.nll_loss_dpred")
    dy = op_utils.to_tensor(dy)
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [dy, y_true, y_pred], attrs)

def nll_loss_dtrue(dy, y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.nll_loss_dtrue")
    dy = op_utils.to_tensor(dy)
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [dy, y_true, y_pred], attrs)

def non_max_suppression(data, valid_count, indices, max_output_size, iou_threshold, force_suppress=False, top_k=-1, coord_start=2, score_index=1, id_index=0, return_indices=True, invalid_to_bottom=False, attrs=None):
    op = GetOp("mnm.op.non_max_suppression")
    data = op_utils.to_tensor(data)
    valid_count = op_utils.to_tensor(valid_count)
    indices = op_utils.to_tensor(indices)
    max_output_size = op_utils.to_tensor(max_output_size)
    iou_threshold = op_utils.to_tensor(iou_threshold)
    force_suppress = op_utils.to_bool(force_suppress)
    top_k = op_utils.to_int(top_k)
    coord_start = op_utils.to_int(coord_start)
    score_index = op_utils.to_int(score_index)
    id_index = op_utils.to_int(id_index)
    return_indices = op_utils.to_bool(return_indices)
    invalid_to_bottom = op_utils.to_bool(invalid_to_bottom)
    return relay.Call(op, [data, valid_count, indices, max_output_size, iou_threshold, force_suppress, top_k, coord_start, score_index, id_index, return_indices, invalid_to_bottom], attrs)

def not_equal(x1, x2, attrs=None):
    op = GetOp("mnm.op.not_equal")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def one_hot(indices, on_value, off_value, depth, axis=-1, dtype="int32", device="cpu", attrs=None):
    op = GetOp("mnm.op.one_hot")
    indices = op_utils.to_tensor(indices)
    on_value = op_utils.to_tensor(on_value)
    off_value = op_utils.to_tensor(off_value)
    depth = op_utils.to_int(depth)
    axis = op_utils.to_int(axis)
    dtype = op_utils.to_string(dtype)
    device = op_utils.to_string(device)
    return relay.Call(op, [indices, on_value, off_value, depth, axis, dtype, device], attrs)

def ones(shape, dtype="int32", device="cpu", attrs=None):
    op = GetOp("mnm.op.ones")
    shape = op_utils.to_int_tuple(shape)
    dtype = op_utils.to_string(dtype)
    device = op_utils.to_string(device)
    return relay.Call(op, [shape, dtype, device], attrs)

def ones_like(x, attrs=None):
    op = GetOp("mnm.op.ones_like")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def pad(x, pad_width, pad_value=0.0, pad_mode="constant", attrs=None):
    op = GetOp("mnm.op.pad")
    x = op_utils.to_tensor(x)
    pad_width = op_utils.to_int_tuple(pad_width)
    pad_value = op_utils.to_double(pad_value)
    pad_mode = op_utils.to_string(pad_mode)
    return relay.Call(op, [x, pad_width, pad_value, pad_mode], attrs)

def power(x1, x2, attrs=None):
    op = GetOp("mnm.op.power")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def prod(x, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.prod")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def prod_dx(x, dy, axis=(), keepdims=False, exclude=False, attrs=None):
    op = GetOp("mnm.op.prod_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_bool(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, dy, axis, keepdims, exclude], attrs)

def relu(x, attrs=None):
    op = GetOp("mnm.op.relu")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def relu_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.relu_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def repeat(x, repeats, axis=None, attrs=None):
    op = GetOp("mnm.op.repeat")
    x = op_utils.to_tensor(x)
    repeats = op_utils.to_int(repeats)
    axis = op_utils.to_any(axis)
    return relay.Call(op, [x, repeats, axis], attrs)

def repeat_dx(x, dy, repeats, axis=None, attrs=None):
    op = GetOp("mnm.op.repeat_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    repeats = op_utils.to_int(repeats)
    axis = op_utils.to_any(axis)
    return relay.Call(op, [x, dy, repeats, axis], attrs)

def reshape(x, shape, reverse=False, attrs=None):
    op = GetOp("mnm.op.reshape")
    x = op_utils.to_tensor(x)
    shape = op_utils.to_int_tuple(shape)
    reverse = op_utils.to_bool(reverse)
    return relay.Call(op, [x, shape, reverse], attrs)

def resize2d(x, size, layout="NCHW", method="linear", coordinate_transformation_mode="half_pixel", rounding_method="", cubic_alpha=-0.5, cubic_exclude=0, out_dtype="", attrs=None):
    op = GetOp("mnm.op.resize2d")
    x = op_utils.to_tensor(x)
    size = op_utils.to_int_tuple(size)
    layout = op_utils.to_string(layout)
    method = op_utils.to_string(method)
    coordinate_transformation_mode = op_utils.to_string(coordinate_transformation_mode)
    rounding_method = op_utils.to_string(rounding_method)
    cubic_alpha = op_utils.to_double(cubic_alpha)
    cubic_exclude = op_utils.to_int(cubic_exclude)
    out_dtype = op_utils.to_string(out_dtype)
    return relay.Call(op, [x, size, layout, method, coordinate_transformation_mode, rounding_method, cubic_alpha, cubic_exclude, out_dtype], attrs)

def resize2d_dx(x, dy, size, layout="NCHW", method="linear", coordinate_transformation_mode="half_pixel", rounding_method="", cubic_alpha=-0.5, cubic_exclude=0, out_dtype="", attrs=None):
    op = GetOp("mnm.op.resize2d_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    size = op_utils.to_int_tuple(size)
    layout = op_utils.to_string(layout)
    method = op_utils.to_string(method)
    coordinate_transformation_mode = op_utils.to_string(coordinate_transformation_mode)
    rounding_method = op_utils.to_string(rounding_method)
    cubic_alpha = op_utils.to_double(cubic_alpha)
    cubic_exclude = op_utils.to_int(cubic_exclude)
    out_dtype = op_utils.to_string(out_dtype)
    return relay.Call(op, [x, dy, size, layout, method, coordinate_transformation_mode, rounding_method, cubic_alpha, cubic_exclude, out_dtype], attrs)

def reverse(x, axis=0, attrs=None):
    op = GetOp("mnm.op.reverse")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def reverse_sequence(x, sequence_length, seq_axis=1, batch_axis=0, attrs=None):
    op = GetOp("mnm.op.reverse_sequence")
    x = op_utils.to_tensor(x)
    sequence_length = op_utils.to_tensor(sequence_length)
    seq_axis = op_utils.to_int(seq_axis)
    batch_axis = op_utils.to_int(batch_axis)
    return relay.Call(op, [x, sequence_length, seq_axis, batch_axis], attrs)

def right_shift(x1, x2, attrs=None):
    op = GetOp("mnm.op.right_shift")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    return relay.Call(op, [x1, x2], attrs)

def roi_align(data, rois, pooled_size, spatial_scale, sample_ratio=-1, layout="NCHW", mode="avg", attrs=None):
    op = GetOp("mnm.op.roi_align")
    data = op_utils.to_tensor(data)
    rois = op_utils.to_tensor(rois)
    pooled_size = op_utils.to_int_tuple(pooled_size)
    spatial_scale = op_utils.to_double(spatial_scale)
    sample_ratio = op_utils.to_int(sample_ratio)
    layout = op_utils.to_string(layout)
    mode = op_utils.to_string(mode)
    return relay.Call(op, [data, rois, pooled_size, spatial_scale, sample_ratio, layout, mode], attrs)

def roi_align_dx(data, rois, dy, pooled_size, spatial_scale, sample_ratio=-1, layout="NCHW", mode="avg", attrs=None):
    op = GetOp("mnm.op.roi_align_dx")
    data = op_utils.to_tensor(data)
    rois = op_utils.to_tensor(rois)
    dy = op_utils.to_tensor(dy)
    pooled_size = op_utils.to_int_tuple(pooled_size)
    spatial_scale = op_utils.to_double(spatial_scale)
    sample_ratio = op_utils.to_int(sample_ratio)
    layout = op_utils.to_string(layout)
    mode = op_utils.to_string(mode)
    return relay.Call(op, [data, rois, dy, pooled_size, spatial_scale, sample_ratio, layout, mode], attrs)

def round(x, attrs=None):
    op = GetOp("mnm.op.round")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def rsqrt(x, attrs=None):
    op = GetOp("mnm.op.rsqrt")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def scatter(x, index, src, axis, attrs=None):
    op = GetOp("mnm.op.scatter")
    x = op_utils.to_tensor(x)
    index = op_utils.to_tensor(index)
    src = op_utils.to_tensor(src)
    axis = op_utils.to_any(axis)
    return relay.Call(op, [x, index, src, axis], attrs)

def scatter_dx(x, y, dy, index, src, axis, attrs=None):
    op = GetOp("mnm.op.scatter_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    index = op_utils.to_tensor(index)
    src = op_utils.to_tensor(src)
    axis = op_utils.to_any(axis)
    return relay.Call(op, [x, y, dy, index, src, axis], attrs)

def sequence_mask(x, sequence_length, mask_value=0.0, axis=0, attrs=None):
    op = GetOp("mnm.op.sequence_mask")
    x = op_utils.to_tensor(x)
    sequence_length = op_utils.to_tensor(sequence_length)
    mask_value = op_utils.to_double(mask_value)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, sequence_length, mask_value, axis], attrs)

def set_stream(device_id, stream_id, attrs=None):
    op = GetOp("mnm.op.set_stream")
    device_id = op_utils.to_int(device_id)
    stream_id = op_utils.to_int(stream_id)
    return relay.Call(op, [device_id, stream_id], attrs)

def sgd(x, dx, v, learning_rate, mu, attrs=None):
    op = GetOp("mnm.op.sgd")
    x = op_utils.to_tensor(x)
    dx = op_utils.to_tensor(dx)
    v = op_utils.to_tensor(v)
    learning_rate = op_utils.to_double(learning_rate)
    mu = op_utils.to_double(mu)
    return relay.Call(op, [x, dx, v, learning_rate, mu], attrs)

def shape(x, attrs=None):
    op = GetOp("mnm.op.shape")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def sigmoid(x, attrs=None):
    op = GetOp("mnm.op.sigmoid")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def sigmoid_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.sigmoid_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def sign(x, attrs=None):
    op = GetOp("mnm.op.sign")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def sin(x, attrs=None):
    op = GetOp("mnm.op.sin")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def smooth_l1_loss(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.smooth_l1_loss")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def smooth_l1_loss_dpred(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.smooth_l1_loss_dpred")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def smooth_l1_loss_dtrue(y_true, y_pred, attrs=None):
    op = GetOp("mnm.op.smooth_l1_loss_dtrue")
    y_true = op_utils.to_tensor(y_true)
    y_pred = op_utils.to_tensor(y_pred)
    return relay.Call(op, [y_true, y_pred], attrs)

def softmax(x, axis=-1, attrs=None):
    op = GetOp("mnm.op.softmax")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def softmax_dx(x, y, dy, axis=-1, attrs=None):
    op = GetOp("mnm.op.softmax_dx")
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, y, dy, axis], attrs)

def sort(data, axis=-1, is_ascend=True, attrs=None):
    op = GetOp("mnm.op.sort")
    data = op_utils.to_tensor(data)
    axis = op_utils.to_int(axis)
    is_ascend = op_utils.to_bool(is_ascend)
    return relay.Call(op, [data, axis, is_ascend], attrs)

def split(x, indices_or_sections=None, axis=0, attrs=None):
    op = GetOp("mnm.op.split")
    x = op_utils.to_tensor(x)
    indices_or_sections = op_utils.to_any(indices_or_sections)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, indices_or_sections, axis], attrs)

def sqrt(x, attrs=None):
    op = GetOp("mnm.op.sqrt")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def sqrt_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.sqrt_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def squeeze(x, axis=None, attrs=None):
    op = GetOp("mnm.op.squeeze")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    return relay.Call(op, [x, axis], attrs)

def stack(x, axis=0, attrs=None):
    op = GetOp("mnm.op.stack")
    x = op_utils.to_tensor_tuple(x)
    axis = op_utils.to_int(axis)
    return relay.Call(op, [x, axis], attrs)

def stream_barrier(attrs=None):
    op = GetOp("mnm.op.stream_barrier")

    return relay.Call(op, [], attrs)

def stream_sync(x, stream_tag=0, attrs=None):
    op = GetOp("mnm.op.stream_sync")
    x = op_utils.to_tensor(x)
    stream_tag = op_utils.to_int(stream_tag)
    return relay.Call(op, [x, stream_tag], attrs)

def strided_slice(x, begin, end, strides=None, slice_mode="end", attrs=None):
    op = GetOp("mnm.op.strided_slice")
    x = op_utils.to_tensor(x)
    begin = op_utils.to_int_tuple(begin)
    end = op_utils.to_int_tuple(end)
    strides = op_utils.to_int_tuple(strides)
    slice_mode = op_utils.to_string(slice_mode)
    return relay.Call(op, [x, begin, end, strides, slice_mode], attrs)

def strided_slice_dx(dy, primal_shape, begin, end, strides=None, slice_mode="end", attrs=None):
    op = GetOp("mnm.op.strided_slice_dx")
    dy = op_utils.to_tensor(dy)
    primal_shape = op_utils.to_int_tuple(primal_shape)
    begin = op_utils.to_int_tuple(begin)
    end = op_utils.to_int_tuple(end)
    strides = op_utils.to_int_tuple(strides)
    slice_mode = op_utils.to_string(slice_mode)
    return relay.Call(op, [dy, primal_shape, begin, end, strides, slice_mode], attrs)

def subtract(x1, x2, out=None, where=None, attrs=None):
    op = GetOp("mnm.op.subtract")
    x1 = op_utils.to_any(x1)
    x2 = op_utils.to_any(x2)
    out = op_utils.to_any(out)
    where = op_utils.to_any(where)
    return relay.Call(op, [x1, x2, out, where], attrs)

def sum(x, axis=(), keepdims=0, exclude=False, attrs=None):
    op = GetOp("mnm.op.sum")
    x = op_utils.to_tensor(x)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_int_tuple(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, axis, keepdims, exclude], attrs)

def sum_dx(x, dy, axis=(), keepdims=0, exclude=False, attrs=None):
    op = GetOp("mnm.op.sum_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    axis = op_utils.to_int_tuple(axis)
    keepdims = op_utils.to_int_tuple(keepdims)
    exclude = op_utils.to_bool(exclude)
    return relay.Call(op, [x, dy, axis, keepdims, exclude], attrs)

def swap_axis(x, axis1, axis2, attrs=None):
    op = GetOp("mnm.op.swap_axis")
    x = op_utils.to_tensor(x)
    axis1 = op_utils.to_int(axis1)
    axis2 = op_utils.to_int(axis2)
    return relay.Call(op, [x, axis1, axis2], attrs)

def take(x, indices, axis=None, mode="clip", attrs=None):
    op = GetOp("mnm.op.take")
    x = op_utils.to_tensor(x)
    indices = op_utils.to_tensor(indices)
    axis = op_utils.to_any(axis)
    mode = op_utils.to_string(mode)
    return relay.Call(op, [x, indices, axis, mode], attrs)

def take_dx(x, dy, indices, axis=None, mode="clip", attrs=None):
    op = GetOp("mnm.op.take_dx")
    x = op_utils.to_tensor(x)
    dy = op_utils.to_tensor(dy)
    indices = op_utils.to_tensor(indices)
    axis = op_utils.to_any(axis)
    mode = op_utils.to_string(mode)
    return relay.Call(op, [x, dy, indices, axis, mode], attrs)

def tanh(x, attrs=None):
    op = GetOp("mnm.op.tanh")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def tanh_dx(x, y, dy, attrs=None):
    op = GetOp("mnm.op.tanh_dx")
    x = op_utils.to_any(x)
    y = op_utils.to_tensor(y)
    dy = op_utils.to_tensor(dy)
    return relay.Call(op, [x, y, dy], attrs)

def threefry_generate(key, shape, attrs=None):
    op = GetOp("mnm.op.threefry_generate")
    key = op_utils.to_tensor(key)
    shape = op_utils.to_int_tuple(shape)
    return relay.Call(op, [key, shape], attrs)

def threefry_split(key, attrs=None):
    op = GetOp("mnm.op.threefry_split")
    key = op_utils.to_tensor(key)
    return relay.Call(op, [key], attrs)

def threshold(x, threshold=0.0, value=0.0, attrs=None):
    op = GetOp("mnm.op.threshold")
    x = op_utils.to_any(x)
    threshold = op_utils.to_double(threshold)
    value = op_utils.to_double(value)
    return relay.Call(op, [x, threshold, value], attrs)

def threshold_dx(x, dy, threshold=0.0, attrs=None):
    op = GetOp("mnm.op.threshold_dx")
    x = op_utils.to_any(x)
    dy = op_utils.to_tensor(dy)
    threshold = op_utils.to_double(threshold)
    return relay.Call(op, [x, dy, threshold], attrs)

def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64", attrs=None):
    op = GetOp("mnm.op.topk")
    data = op_utils.to_tensor(data)
    k = op_utils.to_int(k)
    axis = op_utils.to_int(axis)
    ret_type = op_utils.to_string(ret_type)
    is_ascend = op_utils.to_bool(is_ascend)
    dtype = op_utils.to_string(dtype)
    return relay.Call(op, [data, k, axis, ret_type, is_ascend, dtype], attrs)

def transpose(x, axes=None, attrs=None):
    op = GetOp("mnm.op.transpose")
    x = op_utils.to_tensor(x)
    axes = op_utils.to_int_tuple(axes)
    return relay.Call(op, [x, axes], attrs)

def transpose_dx(dy, axes=None, primal_shape=None, attrs=None):
    op = GetOp("mnm.op.transpose_dx")
    dy = op_utils.to_tensor(dy)
    axes = op_utils.to_int_tuple(axes)
    primal_shape = op_utils.to_int_tuple(primal_shape)
    return relay.Call(op, [dy, axes, primal_shape], attrs)

def trunc(x, attrs=None):
    op = GetOp("mnm.op.trunc")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)

def upper_bound_argwhere(condition, attrs=None):
    op = GetOp("mnm.op.upper_bound.argwhere")
    condition = op_utils.to_tensor(condition)
    return relay.Call(op, [condition], attrs)

def vm_alloc_storage(size, alignment, device_type, device_id, dtype="float32", attrs=None):
    op = GetOp("mnm.op.vm.alloc_storage")
    size = op_utils.to_any(size)
    alignment = op_utils.to_any(alignment)
    device_type = op_utils.to_int(device_type)
    device_id = op_utils.to_int(device_id)
    dtype = op_utils.to_string(dtype)
    return relay.Call(op, [size, alignment, device_type, device_id, dtype], attrs)

def vm_alloc_tensor(storage, shape, dtype="float32", assert_shape=None, own=True, attrs=None):
    op = GetOp("mnm.op.vm.alloc_tensor")
    storage = op_utils.to_tensor(storage)
    shape = op_utils.to_any(shape)
    dtype = op_utils.to_string(dtype)
    assert_shape = op_utils.to_int_tuple(assert_shape)
    own = op_utils.to_bool(own)
    return relay.Call(op, [storage, shape, dtype, assert_shape, own], attrs)

def vm_free(memory, attrs=None):
    op = GetOp("mnm.op.vm.free")
    memory = op_utils.to_tensor(memory)
    return relay.Call(op, [memory], attrs)

def vm_infer_type(func, inputs, attrs=None):
    op = GetOp("mnm.op.vm.infer_type")
    func = op_utils.to_any(func)
    inputs = op_utils.to_any(inputs)
    return relay.Call(op, [func, inputs], attrs)

def vm_invoke_op(func, inputs, outputs, attrs=None):
    op = GetOp("mnm.op.vm.invoke_op")
    func = op_utils.to_any(func)
    inputs = op_utils.to_any(inputs)
    outputs = op_utils.to_any(outputs)
    return relay.Call(op, [func, inputs, outputs], attrs)

def vm_set_shape(data, shape, attrs=None):
    op = GetOp("mnm.op.vm.set_shape")
    data = op_utils.to_tensor(data)
    shape = op_utils.to_any(shape)
    return relay.Call(op, [data, shape], attrs)

def wait_event(event_id, stream_id=-1, attrs=None):
    op = GetOp("mnm.op.wait_event")
    event_id = op_utils.to_int(event_id)
    stream_id = op_utils.to_int(stream_id)
    return relay.Call(op, [event_id, stream_id], attrs)

def where(condition, x, y, attrs=None):
    op = GetOp("mnm.op.where")
    condition = op_utils.to_tensor(condition)
    x = op_utils.to_tensor(x)
    y = op_utils.to_tensor(y)
    return relay.Call(op, [condition, x, y], attrs)

def zeros(shape, dtype="int32", device="cpu", attrs=None):
    op = GetOp("mnm.op.zeros")
    shape = op_utils.to_int_tuple(shape)
    dtype = op_utils.to_string(dtype)
    device = op_utils.to_string(device)
    return relay.Call(op, [shape, dtype, device], attrs)

def zeros_like(x, attrs=None):
    op = GetOp("mnm.op.zeros_like")
    x = op_utils.to_any(x)
    return relay.Call(op, [x], attrs)
