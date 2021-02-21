# pylint: disable=invalid-name,line-too-long,too-many-arguments,redefined-builtin,redefined-outer-name
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=protected-access
"""Auto generated. Do not touch."""
import mnm._ffi.op.sym as ffi
from mnm._core.ndarray import Symbol
from . import sym_utils

__all__ = [
    "_allreduce", "abs", "add", "all", "any",
    "argmax", "argmin", "argsort", "atan", "avg_pool2d",
    "avg_pool2d_dx", "batch_flatten", "batch_matmul", "batch_norm_infer", "batch_norm_train",
    "batch_norm_train_dxwb", "bias_add", "broadcast_to", "broadcast_to_like", "cast",
    "cast_like", "ceil", "clip", "clip_dx", "collapse_sum_like",
    "compiler_begin", "compiler_end", "concatenate", "concatenate_dx", "conv2d",
    "conv2d_dw", "conv2d_dx", "copy", "cos", "cross_entropy",
    "cross_entropy_dpred", "cross_entropy_dtrue", "dense", "device_copy", "divide",
    "equal", "erf", "erf_dx", "exp", "expand_dims",
    "floor", "full", "gather", "gather_dx", "gather_nd",
    "gather_nd_dx", "get_kept_dims", "get_reduce_axis", "get_valid_counts", "greater",
    "greater_equal", "layer_norm", "layer_norm_dx", "less", "less_equal",
    "log", "log_softmax", "log_softmax_dx", "logical_and", "logical_not",
    "matmul", "matmul_nt", "matmul_tn", "matmul_tt", "max",
    "max_pool2d", "max_pool2d_dx", "maximum", "mean", "mean_dx",
    "min", "minimum", "mod", "multiply", "negative",
    "nll_loss", "nll_loss_dpred", "nll_loss_dtrue", "non_max_suppression", "not_equal",
    "pad", "power", "prod", "relu", "relu_dx",
    "repeat", "reshape", "reverse", "reverse_sequence", "round",
    "rsqrt", "sequence_mask", "sgd", "shape", "sigmoid",
    "sigmoid_dx", "sign", "sin", "smooth_l1_loss", "smooth_l1_loss_dpred",
    "smooth_l1_loss_dtrue", "softmax", "softmax_dx", "sort", "split",
    "sqrt", "sqrt_dx", "squeeze", "stack", "stack_dx",
    "stream_sync", "strided_slice", "subtract", "sum", "take",
    "take_dx", "tanh", "tanh_dx", "transpose", "transpose_dx",
    "where",
]

def _allreduce(x):
    x = sym_utils.to_tensor_tuple(x)
    return Symbol.from_expr(ffi._allreduce(x))

def abs(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.abs(x))

def add(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.add(x1, x2, out, where))

def all(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.all(x, axis, keepdims))

def any(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.any(x, axis, keepdims))

def argmax(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.argmax(x, axis, keepdims))

def argmin(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.argmin(x, axis, keepdims))

def argsort(data, axis=-1, is_ascend=True, dtype="int32"):
    data = sym_utils.to_tensor(data)
    axis = sym_utils.to_int(axis)
    is_ascend = sym_utils.to_bool(is_ascend)
    dtype = sym_utils.to_string(dtype)
    return Symbol.from_expr(ffi.argsort(data, axis, is_ascend, dtype))

def atan(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.atan(x))

def avg_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True, layout="NCHW"):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    layout = sym_utils.to_string(layout)
    return Symbol.from_expr(ffi.avg_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad, layout))

def avg_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    return Symbol.from_expr(ffi.avg_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad))

def batch_flatten(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.batch_flatten(x))

def batch_matmul(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.batch_matmul(x1, x2))

def batch_norm_infer(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05):
    x = sym_utils.to_tensor(x)
    running_mean = sym_utils.to_tensor(running_mean)
    running_var = sym_utils.to_tensor(running_var)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    momentum = sym_utils.to_double(momentum)
    eps = sym_utils.to_double(eps)
    return Symbol.from_expr(ffi.batch_norm_infer(x, running_mean, running_var, w, b, momentum, eps))

def batch_norm_train(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05):
    x = sym_utils.to_tensor(x)
    running_mean = sym_utils.to_tensor(running_mean)
    running_var = sym_utils.to_tensor(running_var)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    momentum = sym_utils.to_double(momentum)
    eps = sym_utils.to_double(eps)
    return Symbol.from_expr(ffi.batch_norm_train(x, running_mean, running_var, w, b, momentum, eps))

def batch_norm_train_dxwb(dy, x, w, b, eps):
    dy = sym_utils.to_tensor(dy)
    x = sym_utils.to_tensor(x)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    eps = sym_utils.to_double(eps)
    return Symbol.from_expr(ffi.batch_norm_train_dxwb(dy, x, w, b, eps))

def bias_add(x, bias, axis=1):
    x = sym_utils.to_tensor(x)
    bias = sym_utils.to_tensor(bias)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.bias_add(x, bias, axis))

def broadcast_to(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.broadcast_to(x, shape))

def broadcast_to_like(x, broadcast_type):
    x = sym_utils.to_tensor(x)
    broadcast_type = sym_utils.to_tensor(broadcast_type)
    return Symbol.from_expr(ffi.broadcast_to_like(x, broadcast_type))

def cast(data, dtype):
    data = sym_utils.to_tensor(data)
    dtype = sym_utils.to_string(dtype)
    return Symbol.from_expr(ffi.cast(data, dtype))

def cast_like(data, dtype_like):
    data = sym_utils.to_tensor(data)
    dtype_like = sym_utils.to_tensor(dtype_like)
    return Symbol.from_expr(ffi.cast_like(data, dtype_like))

def ceil(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.ceil(x))

def clip(x, a_min, a_max):
    x = sym_utils.to_tensor(x)
    a_min = sym_utils.to_double(a_min)
    a_max = sym_utils.to_double(a_max)
    return Symbol.from_expr(ffi.clip(x, a_min, a_max))

def clip_dx(x, dy, a_min, a_max):
    x = sym_utils.to_tensor(x)
    dy = sym_utils.to_tensor(dy)
    a_min = sym_utils.to_double(a_min)
    a_max = sym_utils.to_double(a_max)
    return Symbol.from_expr(ffi.clip_dx(x, dy, a_min, a_max))

def collapse_sum_like(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.collapse_sum_like(x, shape))

def compiler_begin(x, compiler):
    x = sym_utils.to_tensor(x)
    compiler = sym_utils.to_string(compiler)
    return Symbol.from_expr(ffi.compiler_begin(x, compiler))

def compiler_end(x, compiler):
    x = sym_utils.to_tensor(x)
    compiler = sym_utils.to_string(compiler)
    return Symbol.from_expr(ffi.compiler_end(x, compiler))

def concatenate(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.concatenate(x, axis))

def concatenate_dx(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.concatenate_dx(x, axis))

def conv2d(x, w, stride=1, padding=0, dilation=1, groups=1, layout="NCHW", kernel_layout="OIHW", out_layout="NCHW"):
    x = sym_utils.to_tensor(x)
    w = sym_utils.to_tensor(w)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    layout = sym_utils.to_string(layout)
    kernel_layout = sym_utils.to_string(kernel_layout)
    out_layout = sym_utils.to_string(out_layout)
    return Symbol.from_expr(ffi.conv2d(x, w, stride, padding, dilation, groups, layout, kernel_layout, out_layout))

def conv2d_dw(x_or_w, y, dy, shape, stride, padding, dilation, groups):
    x_or_w = sym_utils.to_tensor(x_or_w)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    shape = sym_utils.to_int_tuple(shape)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d_dw(x_or_w, y, dy, shape, stride, padding, dilation, groups))

def conv2d_dx(x_or_w, y, dy, shape, stride, padding, dilation, groups):
    x_or_w = sym_utils.to_tensor(x_or_w)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    shape = sym_utils.to_int_tuple(shape)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d_dx(x_or_w, y, dy, shape, stride, padding, dilation, groups))

def copy(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.copy(x))

def cos(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.cos(x))

def cross_entropy(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.cross_entropy(y_true, y_pred))

def cross_entropy_dpred(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.cross_entropy_dpred(y_true, y_pred))

def cross_entropy_dtrue(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.cross_entropy_dtrue(y_true, y_pred))

def dense(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.dense(x1, x2))

def device_copy(data, src_dev_type=0, dst_dev_type=0):
    data = sym_utils.to_tensor(data)
    src_dev_type = sym_utils.to_int(src_dev_type)
    dst_dev_type = sym_utils.to_int(dst_dev_type)
    return Symbol.from_expr(ffi.device_copy(data, src_dev_type, dst_dev_type))

def divide(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.divide(x1, x2, out, where))

def equal(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.equal(x1, x2, out, where))

def erf(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.erf(x))

def erf_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.erf_dx(x, y, dy))

def exp(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.exp(x))

def expand_dims(x, axis, num_newaxis=1):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    num_newaxis = sym_utils.to_int(num_newaxis)
    return Symbol.from_expr(ffi.expand_dims(x, axis, num_newaxis))

def floor(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.floor(x))

def full(fill_value, shape, dtype="int32"):
    fill_value = sym_utils.to_tensor(fill_value)
    shape = sym_utils.to_int_tuple(shape)
    dtype = sym_utils.to_string(dtype)
    return Symbol.from_expr(ffi.full(fill_value, shape, dtype))

def gather(data, axis, indices):
    data = sym_utils.to_tensor(data)
    axis = sym_utils.to_int(axis)
    indices = sym_utils.to_tensor(indices)
    return Symbol.from_expr(ffi.gather(data, axis, indices))

def gather_dx(data, axis, indices, dy):
    data = sym_utils.to_tensor(data)
    axis = sym_utils.to_int(axis)
    indices = sym_utils.to_tensor(indices)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.gather_dx(data, axis, indices, dy))

def gather_nd(data, indices):
    data = sym_utils.to_tensor(data)
    indices = sym_utils.to_tensor(indices)
    return Symbol.from_expr(ffi.gather_nd(data, indices))

def gather_nd_dx(data, indices, dy):
    data = sym_utils.to_tensor(data)
    indices = sym_utils.to_tensor(indices)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.gather_nd_dx(data, indices, dy))

def get_kept_dims(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.get_kept_dims(x1, x2))

def get_reduce_axis(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.get_reduce_axis(x1, x2))

def get_valid_counts(data, score_threshold, id_index=0, score_index=1):
    data = sym_utils.to_tensor(data)
    score_threshold = sym_utils.to_tensor(score_threshold)
    id_index = sym_utils.to_int(id_index)
    score_index = sym_utils.to_int(score_index)
    return Symbol.from_expr(ffi.get_valid_counts(data, score_threshold, id_index, score_index))

def greater(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.greater(x1, x2, out, where))

def greater_equal(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.greater_equal(x1, x2, out, where))

def layer_norm(x, axis=-1, eps=1e-05):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    eps = sym_utils.to_double(eps)
    return Symbol.from_expr(ffi.layer_norm(x, axis, eps))

def layer_norm_dx(x, y, dy, axis=-1, eps=1e-05):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    axis = sym_utils.to_int(axis)
    eps = sym_utils.to_double(eps)
    return Symbol.from_expr(ffi.layer_norm_dx(x, y, dy, axis, eps))

def less(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.less(x1, x2, out, where))

def less_equal(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.less_equal(x1, x2, out, where))

def log(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.log(x))

def log_softmax(x, axis=-1):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.log_softmax(x, axis))

def log_softmax_dx(x, y, dy, axis=-1):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.log_softmax_dx(x, y, dy, axis))

def logical_and(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.logical_and(x1, x2, out, where))

def logical_not(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.logical_not(x))

def matmul(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.matmul(x1, x2))

def matmul_nt(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.matmul_nt(x1, x2))

def matmul_tn(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.matmul_tn(x1, x2))

def matmul_tt(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.matmul_tt(x1, x2))

def max(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.max(x, axis, keepdims))

def max_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True, layout="NCHW"):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    layout = sym_utils.to_string(layout)
    return Symbol.from_expr(ffi.max_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad, layout))

def max_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    return Symbol.from_expr(ffi.max_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad))

def maximum(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.maximum(x1, x2, out, where))

def mean(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.mean(x, axis, keepdims))

def mean_dx(x, y, dy, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.mean_dx(x, y, dy, axis, keepdims))

def min(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.min(x, axis, keepdims))

def minimum(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.minimum(x1, x2, out, where))

def mod(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.mod(x1, x2, out, where))

def multiply(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.multiply(x1, x2, out, where))

def negative(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.negative(x))

def nll_loss(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss(y_true, y_pred))

def nll_loss_dpred(dy, y_true, y_pred):
    dy = sym_utils.to_tensor(dy)
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dpred(dy, y_true, y_pred))

def nll_loss_dtrue(dy, y_true, y_pred):
    dy = sym_utils.to_tensor(dy)
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dtrue(dy, y_true, y_pred))

def non_max_suppression(data, valid_count, indices, max_output_size, iou_threshold, force_suppress=False, top_k=-1, coord_start=2, score_index=1, id_index=0, return_indices=True, invalid_to_bottom=False):
    data = sym_utils.to_tensor(data)
    valid_count = sym_utils.to_tensor(valid_count)
    indices = sym_utils.to_tensor(indices)
    max_output_size = sym_utils.to_tensor(max_output_size)
    iou_threshold = sym_utils.to_tensor(iou_threshold)
    force_suppress = sym_utils.to_bool(force_suppress)
    top_k = sym_utils.to_int(top_k)
    coord_start = sym_utils.to_int(coord_start)
    score_index = sym_utils.to_int(score_index)
    id_index = sym_utils.to_int(id_index)
    return_indices = sym_utils.to_bool(return_indices)
    invalid_to_bottom = sym_utils.to_bool(invalid_to_bottom)
    return Symbol.from_expr(ffi.non_max_suppression(data, valid_count, indices, max_output_size, iou_threshold, force_suppress, top_k, coord_start, score_index, id_index, return_indices, invalid_to_bottom))

def not_equal(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.not_equal(x1, x2, out, where))

def pad(x, pad_width, pad_value=0.0, pad_mode="constant"):
    x = sym_utils.to_tensor(x)
    pad_width = sym_utils.to_int_tuple(pad_width)
    pad_value = sym_utils.to_double(pad_value)
    pad_mode = sym_utils.to_string(pad_mode)
    return Symbol.from_expr(ffi.pad(x, pad_width, pad_value, pad_mode))

def power(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.power(x1, x2, out, where))

def prod(x, axis=(), keepdims=False):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_bool(keepdims)
    return Symbol.from_expr(ffi.prod(x, axis, keepdims))

def relu(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.relu(x))

def relu_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.relu_dx(x, y, dy))

def repeat(x, repeats, axis=None):
    x = sym_utils.to_tensor(x)
    repeats = sym_utils.to_int(repeats)
    axis = sym_utils.to_any(axis)
    return Symbol.from_expr(ffi.repeat(x, repeats, axis))

def reshape(x, shape, reverse=False):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    reverse = sym_utils.to_bool(reverse)
    return Symbol.from_expr(ffi.reshape(x, shape, reverse))

def reverse(x, axis=0):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.reverse(x, axis))

def reverse_sequence(x, sequence_length, seq_axis=1, batch_axis=0):
    x = sym_utils.to_tensor(x)
    sequence_length = sym_utils.to_tensor(sequence_length)
    seq_axis = sym_utils.to_int(seq_axis)
    batch_axis = sym_utils.to_int(batch_axis)
    return Symbol.from_expr(ffi.reverse_sequence(x, sequence_length, seq_axis, batch_axis))

def round(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.round(x))

def rsqrt(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.rsqrt(x))

def sequence_mask(x, sequence_length, mask_value=0.0, axis=0):
    x = sym_utils.to_tensor(x)
    sequence_length = sym_utils.to_tensor(sequence_length)
    mask_value = sym_utils.to_double(mask_value)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.sequence_mask(x, sequence_length, mask_value, axis))

def sgd(x, dx, v, learning_rate, mu):
    x = sym_utils.to_tensor(x)
    dx = sym_utils.to_tensor(dx)
    v = sym_utils.to_tensor(v)
    learning_rate = sym_utils.to_double(learning_rate)
    mu = sym_utils.to_double(mu)
    return Symbol.from_expr(ffi.sgd(x, dx, v, learning_rate, mu))

def shape(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.shape(x))

def sigmoid(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.sigmoid(x))

def sigmoid_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.sigmoid_dx(x, y, dy))

def sign(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.sign(x))

def sin(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.sin(x))

def smooth_l1_loss(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.smooth_l1_loss(y_true, y_pred))

def smooth_l1_loss_dpred(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.smooth_l1_loss_dpred(y_true, y_pred))

def smooth_l1_loss_dtrue(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.smooth_l1_loss_dtrue(y_true, y_pred))

def softmax(x, axis=-1):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.softmax(x, axis))

def softmax_dx(x, y, dy, axis=-1):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.softmax_dx(x, y, dy, axis))

def sort(data, axis=-1, is_ascend=True):
    data = sym_utils.to_tensor(data)
    axis = sym_utils.to_int(axis)
    is_ascend = sym_utils.to_bool(is_ascend)
    return Symbol.from_expr(ffi.sort(data, axis, is_ascend))

def split(x, indices_or_sections=None, axis=0):
    x = sym_utils.to_tensor(x)
    indices_or_sections = sym_utils.to_any(indices_or_sections)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.split(x, indices_or_sections, axis))

def sqrt(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.sqrt(x))

def sqrt_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.sqrt_dx(x, y, dy))

def squeeze(x, axis):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    return Symbol.from_expr(ffi.squeeze(x, axis))

def stack(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.stack(x, axis))

def stack_dx(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.stack_dx(x, axis))

def stream_sync(x, stream_tag=0):
    x = sym_utils.to_tensor(x)
    stream_tag = sym_utils.to_int(stream_tag)
    return Symbol.from_expr(ffi.stream_sync(x, stream_tag))

def strided_slice(x, begin, end, strides=None, slice_mode="end"):
    x = sym_utils.to_tensor(x)
    begin = sym_utils.to_int_tuple(begin)
    end = sym_utils.to_int_tuple(end)
    strides = sym_utils.to_int_tuple(strides)
    slice_mode = sym_utils.to_string(slice_mode)
    return Symbol.from_expr(ffi.strided_slice(x, begin, end, strides, slice_mode))

def subtract(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.subtract(x1, x2, out, where))

def sum(x, axis=(), keepdims=0):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keepdims = sym_utils.to_int_tuple(keepdims)
    return Symbol.from_expr(ffi.sum(x, axis, keepdims))

def take(x, indices, axis=None, mode="clip"):
    x = sym_utils.to_tensor(x)
    indices = sym_utils.to_tensor(indices)
    axis = sym_utils.to_any(axis)
    mode = sym_utils.to_string(mode)
    return Symbol.from_expr(ffi.take(x, indices, axis, mode))

def take_dx(x, y, dy, indices, axis=None, mode="clip"):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    indices = sym_utils.to_tensor(indices)
    axis = sym_utils.to_any(axis)
    mode = sym_utils.to_string(mode)
    return Symbol.from_expr(ffi.take_dx(x, y, dy, indices, axis, mode))

def tanh(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.tanh(x))

def tanh_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.tanh_dx(x, y, dy))

def transpose(x, axes=None):
    x = sym_utils.to_tensor(x)
    axes = sym_utils.to_int_tuple(axes)
    return Symbol.from_expr(ffi.transpose(x, axes))

def transpose_dx(x, y, dy, axes=None):
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    axes = sym_utils.to_int_tuple(axes)
    return Symbol.from_expr(ffi.transpose_dx(x, y, dy, axes))

def where(condition, x, y):
    condition = sym_utils.to_tensor(condition)
    x = sym_utils.to_tensor(x)
    y = sym_utils.to_tensor(y)
    return Symbol.from_expr(ffi.where(condition, x, y))
