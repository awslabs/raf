# pylint: disable=invalid-name,line-too-long
# pylint: disable=too-many-arguments,redefined-builtin,redefined-outer-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Auto generated. Do not touch."""
import mnm._ffi.op.imp as ffi
from mnm._core.core_utils import set_module
from . import imp_utils

__all__ = [
    "abs", "add", "all", "any", "argmax",
    "argmin", "atan", "avg_pool2d", "avg_pool2d_dx", "batch_flatten",
    "batch_matmul", "batch_norm_infer", "batch_norm_train", "batch_norm_train_dxwb", "bias_add",
    "broadcast_to", "broadcast_to_like", "ceil", "clip", "clip_dx",
    "collapse_sum_like", "concatenate", "concatenate_dx", "conv2d", "conv2d_dw",
    "conv2d_dx", "copy", "cos", "dense", "divide",
    "equal", "erf", "erf_dx", "expand_dims", "floor",
    "get_kept_dims", "get_reduce_axis", "get_valid_counts", "greater", "greater_equal",
    "less", "less_equal", "log", "log_softmax", "log_softmax_dx",
    "logical_not", "matmul", "matmul_nt", "matmul_tn", "matmul_tt",
    "max_pool2d", "max_pool2d_dx", "maximum", "mean", "mean_dx",
    "minimum", "mod", "multiply", "negative", "nll_loss",
    "nll_loss_dpred", "nll_loss_dtrue", "non_max_suppression", "not_equal", "relu",
    "relu_dx", "reshape", "sequence_mask", "sgd", "shape",
    "sigmoid", "sigmoid_dx", "softmax", "softmax_dx", "split",
    "sqrt", "sqrt_dx", "subtract", "sum", "take",
    "take_dx", "tanh", "tanh_dx", "transpose", "transpose_dx",
]

@set_module("mnm")
def abs(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.abs(x))

@set_module("mnm")
def add(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.add(x1, x2, out, where))

@set_module("mnm")
def all(x, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.all(x, axis, keepdims))

@set_module("mnm")
def any(x, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.any(x, axis, keepdims))

@set_module("mnm")
def argmax(x, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.argmax(x, axis, keepdims))

@set_module("mnm")
def argmin(x, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.argmin(x, axis, keepdims))

@set_module("mnm")
def atan(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.atan(x))

@set_module("mnm")
def avg_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = imp_utils.to_tensor(x)
    kernel = imp_utils.to_int_tuple(kernel)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    ceil_mode = imp_utils.to_bool(ceil_mode)
    include_pad = imp_utils.to_bool(include_pad)
    return imp_utils.ret(ffi.avg_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad))

@set_module("mnm")
def avg_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    kernel = imp_utils.to_int_tuple(kernel)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    ceil_mode = imp_utils.to_bool(ceil_mode)
    include_pad = imp_utils.to_bool(include_pad)
    return imp_utils.ret(ffi.avg_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad))

@set_module("mnm")
def batch_flatten(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.batch_flatten(x))

@set_module("mnm")
def batch_matmul(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.batch_matmul(x1, x2))

@set_module("mnm")
def batch_norm_infer(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05):
    x = imp_utils.to_tensor(x)
    running_mean = imp_utils.to_tensor(running_mean)
    running_var = imp_utils.to_tensor(running_var)
    w = imp_utils.to_tensor(w)
    b = imp_utils.to_tensor(b)
    momentum = imp_utils.to_double(momentum)
    eps = imp_utils.to_double(eps)
    return imp_utils.ret(ffi.batch_norm_infer(x, running_mean, running_var, w, b, momentum, eps))

@set_module("mnm")
def batch_norm_train(x, running_mean, running_var, w=None, b=None, momentum=0.1, eps=1e-05):
    x = imp_utils.to_tensor(x)
    running_mean = imp_utils.to_tensor(running_mean)
    running_var = imp_utils.to_tensor(running_var)
    w = imp_utils.to_tensor(w)
    b = imp_utils.to_tensor(b)
    momentum = imp_utils.to_double(momentum)
    eps = imp_utils.to_double(eps)
    return imp_utils.ret(ffi.batch_norm_train(x, running_mean, running_var, w, b, momentum, eps))

@set_module("mnm")
def batch_norm_train_dxwb(dy, x, w, b, eps):
    dy = imp_utils.to_tensor(dy)
    x = imp_utils.to_tensor(x)
    w = imp_utils.to_tensor(w)
    b = imp_utils.to_tensor(b)
    eps = imp_utils.to_double(eps)
    return imp_utils.ret(ffi.batch_norm_train_dxwb(dy, x, w, b, eps))

@set_module("mnm")
def bias_add(x, bias, axis=1):
    x = imp_utils.to_tensor(x)
    bias = imp_utils.to_tensor(bias)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.bias_add(x, bias, axis))

@set_module("mnm")
def broadcast_to(x, shape):
    x = imp_utils.to_tensor(x)
    shape = imp_utils.to_int_tuple(shape)
    return imp_utils.ret(ffi.broadcast_to(x, shape))

@set_module("mnm")
def broadcast_to_like(x, broadcast_type):
    x = imp_utils.to_tensor(x)
    broadcast_type = imp_utils.to_tensor(broadcast_type)
    return imp_utils.ret(ffi.broadcast_to_like(x, broadcast_type))

@set_module("mnm")
def ceil(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.ceil(x))

@set_module("mnm")
def clip(x, a_min, a_max):
    x = imp_utils.to_tensor(x)
    a_min = imp_utils.to_double(a_min)
    a_max = imp_utils.to_double(a_max)
    return imp_utils.ret(ffi.clip(x, a_min, a_max))

@set_module("mnm")
def clip_dx(x, dy, a_min, a_max):
    x = imp_utils.to_tensor(x)
    dy = imp_utils.to_tensor(dy)
    a_min = imp_utils.to_double(a_min)
    a_max = imp_utils.to_double(a_max)
    return imp_utils.ret(ffi.clip_dx(x, dy, a_min, a_max))

@set_module("mnm")
def collapse_sum_like(x, shape):
    x = imp_utils.to_tensor(x)
    shape = imp_utils.to_int_tuple(shape)
    return imp_utils.ret(ffi.collapse_sum_like(x, shape))

@set_module("mnm")
def concatenate(x, axis=0):
    x = imp_utils.to_tensor_tuple(x)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.concatenate(x, axis))

@set_module("mnm")
def concatenate_dx(x, axis=0):
    x = imp_utils.to_tensor_tuple(x)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.concatenate_dx(x, axis))

@set_module("mnm")
def conv2d(x, w, stride=1, padding=0, dilation=1, groups=1):
    x = imp_utils.to_tensor(x)
    w = imp_utils.to_tensor(w)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    groups = imp_utils.to_int(groups)
    return imp_utils.ret(ffi.conv2d(x, w, stride, padding, dilation, groups))

@set_module("mnm")
def conv2d_dw(x_or_w, y, dy, shape, stride, padding, dilation, groups):
    x_or_w = imp_utils.to_tensor(x_or_w)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    shape = imp_utils.to_int_tuple(shape)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    groups = imp_utils.to_int(groups)
    return imp_utils.ret(ffi.conv2d_dw(x_or_w, y, dy, shape, stride, padding, dilation, groups))

@set_module("mnm")
def conv2d_dx(x_or_w, y, dy, shape, stride, padding, dilation, groups):
    x_or_w = imp_utils.to_tensor(x_or_w)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    shape = imp_utils.to_int_tuple(shape)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    groups = imp_utils.to_int(groups)
    return imp_utils.ret(ffi.conv2d_dx(x_or_w, y, dy, shape, stride, padding, dilation, groups))

@set_module("mnm")
def copy(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.copy(x))

@set_module("mnm")
def cos(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.cos(x))

@set_module("mnm")
def dense(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.dense(x1, x2))

@set_module("mnm")
def divide(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.divide(x1, x2, out, where))

@set_module("mnm")
def equal(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.equal(x1, x2, out, where))

@set_module("mnm")
def erf(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.erf(x))

@set_module("mnm")
def erf_dx(x, y, dy):
    x = imp_utils.to_any(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    return imp_utils.ret(ffi.erf_dx(x, y, dy))

@set_module("mnm")
def expand_dims(x, axis, num_newaxis=1):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int(axis)
    num_newaxis = imp_utils.to_int(num_newaxis)
    return imp_utils.ret(ffi.expand_dims(x, axis, num_newaxis))

@set_module("mnm")
def floor(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.floor(x))

@set_module("mnm")
def get_kept_dims(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.get_kept_dims(x1, x2))

@set_module("mnm")
def get_reduce_axis(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.get_reduce_axis(x1, x2))

@set_module("mnm")
def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    data = imp_utils.to_tensor(data)
    score_threshold = imp_utils.to_double(score_threshold)
    id_index = imp_utils.to_int(id_index)
    score_index = imp_utils.to_int(score_index)
    return imp_utils.ret(ffi.get_valid_counts(data, score_threshold, id_index, score_index))

@set_module("mnm")
def greater(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.greater(x1, x2, out, where))

@set_module("mnm")
def greater_equal(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.greater_equal(x1, x2, out, where))

@set_module("mnm")
def less(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.less(x1, x2, out, where))

@set_module("mnm")
def less_equal(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.less_equal(x1, x2, out, where))

@set_module("mnm")
def log(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.log(x))

@set_module("mnm")
def log_softmax(x, axis=-1):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.log_softmax(x, axis))

@set_module("mnm")
def log_softmax_dx(x, y, dy, axis=-1):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.log_softmax_dx(x, y, dy, axis))

@set_module("mnm")
def logical_not(x, out=None, where=None):
    x = imp_utils.to_any(x)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.logical_not(x, out, where))

@set_module("mnm")
def matmul(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.matmul(x1, x2))

@set_module("mnm")
def matmul_nt(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.matmul_nt(x1, x2))

@set_module("mnm")
def matmul_tn(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.matmul_tn(x1, x2))

@set_module("mnm")
def matmul_tt(x1, x2):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    return imp_utils.ret(ffi.matmul_tt(x1, x2))

@set_module("mnm")
def max_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = imp_utils.to_tensor(x)
    kernel = imp_utils.to_int_tuple(kernel)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    ceil_mode = imp_utils.to_bool(ceil_mode)
    include_pad = imp_utils.to_bool(include_pad)
    return imp_utils.ret(ffi.max_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad))

@set_module("mnm")
def max_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    kernel = imp_utils.to_int_tuple(kernel)
    stride = imp_utils.to_int_tuple(stride)
    padding = imp_utils.to_int_tuple(padding)
    dilation = imp_utils.to_int_tuple(dilation)
    ceil_mode = imp_utils.to_bool(ceil_mode)
    include_pad = imp_utils.to_bool(include_pad)
    return imp_utils.ret(ffi.max_pool2d_dx(x, y, dy, kernel, stride, padding, dilation, ceil_mode, include_pad))

@set_module("mnm")
def maximum(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.maximum(x1, x2, out, where))

@set_module("mnm")
def mean(x, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.mean(x, axis, keepdims))

@set_module("mnm")
def mean_dx(x, y, dy, axis=(), keepdims=False):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    axis = imp_utils.to_int_tuple(axis)
    keepdims = imp_utils.to_bool(keepdims)
    return imp_utils.ret(ffi.mean_dx(x, y, dy, axis, keepdims))

@set_module("mnm")
def minimum(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.minimum(x1, x2, out, where))

@set_module("mnm")
def mod(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.mod(x1, x2, out, where))

@set_module("mnm")
def multiply(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.multiply(x1, x2, out, where))

@set_module("mnm")
def negative(x, out=None, where=None):
    x = imp_utils.to_any(x)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.negative(x, out, where))

@set_module("mnm")
def nll_loss(y_true, y_pred):
    y_true = imp_utils.to_tensor(y_true)
    y_pred = imp_utils.to_tensor(y_pred)
    return imp_utils.ret(ffi.nll_loss(y_true, y_pred))

@set_module("mnm")
def nll_loss_dpred(y_true, y_pred):
    y_true = imp_utils.to_tensor(y_true)
    y_pred = imp_utils.to_tensor(y_pred)
    return imp_utils.ret(ffi.nll_loss_dpred(y_true, y_pred))

@set_module("mnm")
def nll_loss_dtrue(y_true, y_pred):
    y_true = imp_utils.to_tensor(y_true)
    y_pred = imp_utils.to_tensor(y_pred)
    return imp_utils.ret(ffi.nll_loss_dtrue(y_true, y_pred))

@set_module("mnm")
def non_max_suppression(data, valid_count, indices, max_output_size, iou_threshold=0.5, force_suppress=False, top_k=-1, coord_start=2, score_index=1, id_index=0, return_indices=True, invalid_to_bottom=False):
    data = imp_utils.to_tensor(data)
    valid_count = imp_utils.to_tensor(valid_count)
    indices = imp_utils.to_tensor(indices)
    max_output_size = imp_utils.to_tensor(max_output_size)
    iou_threshold = imp_utils.to_double(iou_threshold)
    force_suppress = imp_utils.to_bool(force_suppress)
    top_k = imp_utils.to_int(top_k)
    coord_start = imp_utils.to_int(coord_start)
    score_index = imp_utils.to_int(score_index)
    id_index = imp_utils.to_int(id_index)
    return_indices = imp_utils.to_bool(return_indices)
    invalid_to_bottom = imp_utils.to_bool(invalid_to_bottom)
    return imp_utils.ret(ffi.non_max_suppression(data, valid_count, indices, max_output_size, iou_threshold, force_suppress, top_k, coord_start, score_index, id_index, return_indices, invalid_to_bottom))

@set_module("mnm")
def not_equal(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.not_equal(x1, x2, out, where))

@set_module("mnm")
def relu(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.relu(x))

@set_module("mnm")
def relu_dx(x, y, dy):
    x = imp_utils.to_any(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    return imp_utils.ret(ffi.relu_dx(x, y, dy))

@set_module("mnm")
def reshape(x, shape, reverse=False):
    x = imp_utils.to_tensor(x)
    shape = imp_utils.to_int_tuple(shape)
    reverse = imp_utils.to_bool(reverse)
    return imp_utils.ret(ffi.reshape(x, shape, reverse))

@set_module("mnm")
def sequence_mask(x, sequence_length, mask_value=0.0, axis=0):
    x = imp_utils.to_tensor(x)
    sequence_length = imp_utils.to_tensor(sequence_length)
    mask_value = imp_utils.to_double(mask_value)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.sequence_mask(x, sequence_length, mask_value, axis))

@set_module("mnm")
def sgd(x, dx, v, learning_rate, mu):
    x = imp_utils.to_tensor(x)
    dx = imp_utils.to_tensor(dx)
    v = imp_utils.to_tensor(v)
    learning_rate = imp_utils.to_double(learning_rate)
    mu = imp_utils.to_double(mu)
    return imp_utils.ret(ffi.sgd(x, dx, v, learning_rate, mu))

@set_module("mnm")
def shape(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.shape(x))

@set_module("mnm")
def sigmoid(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.sigmoid(x))

@set_module("mnm")
def sigmoid_dx(x, y, dy):
    x = imp_utils.to_any(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    return imp_utils.ret(ffi.sigmoid_dx(x, y, dy))

@set_module("mnm")
def softmax(x, axis=-1):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.softmax(x, axis))

@set_module("mnm")
def softmax_dx(x, y, dy, axis=-1):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.softmax_dx(x, y, dy, axis))

@set_module("mnm")
def split(x, indices_or_sections, axis=0):
    x = imp_utils.to_tensor(x)
    indices_or_sections = imp_utils.to_int_tuple(indices_or_sections)
    axis = imp_utils.to_int(axis)
    return imp_utils.ret(ffi.split(x, indices_or_sections, axis))

@set_module("mnm")
def sqrt(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.sqrt(x))

@set_module("mnm")
def sqrt_dx(x, y, dy):
    x = imp_utils.to_any(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    return imp_utils.ret(ffi.sqrt_dx(x, y, dy))

@set_module("mnm")
def subtract(x1, x2, out=None, where=None):
    x1 = imp_utils.to_any(x1)
    x2 = imp_utils.to_any(x2)
    out = imp_utils.to_any(out)
    where = imp_utils.to_any(where)
    return imp_utils.ret(ffi.subtract(x1, x2, out, where))

@set_module("mnm")
def sum(x, axis, keep):
    x = imp_utils.to_tensor(x)
    axis = imp_utils.to_int_tuple(axis)
    keep = imp_utils.to_int_tuple(keep)
    return imp_utils.ret(ffi.sum(x, axis, keep))

@set_module("mnm")
def take(x, indices, axis=None):
    x = imp_utils.to_tensor(x)
    indices = imp_utils.to_tensor(indices)
    axis = imp_utils.to_any(axis)
    return imp_utils.ret(ffi.take(x, indices, axis))

@set_module("mnm")
def take_dx(x, y, dy, indices, axis=None):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    indices = imp_utils.to_tensor(indices)
    axis = imp_utils.to_any(axis)
    return imp_utils.ret(ffi.take_dx(x, y, dy, indices, axis))

@set_module("mnm")
def tanh(x):
    x = imp_utils.to_any(x)
    return imp_utils.ret(ffi.tanh(x))

@set_module("mnm")
def tanh_dx(x, y, dy):
    x = imp_utils.to_any(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    return imp_utils.ret(ffi.tanh_dx(x, y, dy))

@set_module("mnm")
def transpose(x, axes=None):
    x = imp_utils.to_tensor(x)
    axes = imp_utils.to_int_tuple(axes)
    return imp_utils.ret(ffi.transpose(x, axes))

@set_module("mnm")
def transpose_dx(x, y, dy, axes=None):
    x = imp_utils.to_tensor(x)
    y = imp_utils.to_tensor(y)
    dy = imp_utils.to_tensor(dy)
    axes = imp_utils.to_int_tuple(axes)
    return imp_utils.ret(ffi.transpose_dx(x, y, dy, axes))
