# pylint: disable=invalid-name,line-too-long,too-many-arguments,redefined-builtin,redefined-outer-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Auto generated. Do not touch."""
import mnm._ffi.op.sym as ffi
from mnm._core.ndarray import Symbol
from . import sym_utils

__all__ = [
    "abs", "add", "all", "any", "argmax",
    "argmin", "atan", "avg_pool2d", "avg_pool2d_dx", "batch_flatten",
    "batch_matmul", "batch_norm_infer", "batch_norm_train", "batch_norm_train_dxwb", "broadcast_to",
    "broadcast_to_like", "ceil", "clip", "clip_dx", "collapse_sum_like",
    "concatenate", "concatenate_dx", "conv2d", "conv2d_dw", "conv2d_dx",
    "copy", "cos", "divide", "equal", "erf",
    "erf_dx", "expand_dims", "floor", "get_kept_dims", "get_reduce_axis",
    "greater", "greater_equal", "less", "less_equal", "log",
    "log_softmax", "log_softmax_dx", "logical_not", "matmul", "matmul_nt",
    "matmul_tn", "matmul_tt", "max_pool2d", "max_pool2d_dx", "maximum",
    "mean", "mean_dx", "minimum", "mod", "multiply",
    "negative", "nll_loss", "nll_loss_dpred", "nll_loss_dtrue", "not_equal",
    "relu", "relu_dx", "reshape", "reshape_dx", "sequence_mask",
    "sgd", "shape", "sigmoid", "sigmoid_dx", "softmax",
    "softmax_dx", "split", "sqrt", "sqrt_dx", "subtract",
    "sum", "take", "tanh", "tanh_dx", "transpose",
    "transpose_dx",
]

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

def atan(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.atan(x))

def avg_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    return Symbol.from_expr(ffi.avg_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad))

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

def broadcast_to(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.broadcast_to(x, shape))

def broadcast_to_like(x, broadcast_type):
    x = sym_utils.to_tensor(x)
    broadcast_type = sym_utils.to_tensor(broadcast_type)
    return Symbol.from_expr(ffi.broadcast_to_like(x, broadcast_type))

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

def concatenate(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.concatenate(x, axis))

def concatenate_dx(x, axis=0):
    x = sym_utils.to_tensor_tuple(x)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.concatenate_dx(x, axis))

def conv2d(x, w, stride=1, padding=0, dilation=1, groups=1):
    x = sym_utils.to_tensor(x)
    w = sym_utils.to_tensor(w)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d(x, w, stride, padding, dilation, groups))

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

def expand_dims(x, axis, num_newaxis=1):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int(axis)
    num_newaxis = sym_utils.to_int(num_newaxis)
    return Symbol.from_expr(ffi.expand_dims(x, axis, num_newaxis))

def floor(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.floor(x))

def get_kept_dims(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.get_kept_dims(x1, x2))

def get_reduce_axis(x1, x2):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    return Symbol.from_expr(ffi.get_reduce_axis(x1, x2))

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

def logical_not(x, out=None, where=None):
    x = sym_utils.to_any(x)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.logical_not(x, out, where))

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

def max_pool2d(x, kernel, stride, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    ceil_mode = sym_utils.to_bool(ceil_mode)
    include_pad = sym_utils.to_bool(include_pad)
    return Symbol.from_expr(ffi.max_pool2d(x, kernel, stride, padding, dilation, ceil_mode, include_pad))

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

def negative(x, out=None, where=None):
    x = sym_utils.to_any(x)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.negative(x, out, where))

def nll_loss(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss(y_true, y_pred))

def nll_loss_dpred(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dpred(y_true, y_pred))

def nll_loss_dtrue(y_true, y_pred):
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dtrue(y_true, y_pred))

def not_equal(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.not_equal(x1, x2, out, where))

def relu(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.relu(x))

def relu_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.relu_dx(x, y, dy))

def reshape(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.reshape(x, shape))

def reshape_dx(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.reshape_dx(x, shape))

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

def split(x, indices_or_sections, axis=0):
    x = sym_utils.to_tensor(x)
    indices_or_sections = sym_utils.to_int_tuple(indices_or_sections)
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

def subtract(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.subtract(x1, x2, out, where))

def sum(x, axis, keep):
    x = sym_utils.to_tensor(x)
    axis = sym_utils.to_int_tuple(axis)
    keep = sym_utils.to_int_tuple(keep)
    return Symbol.from_expr(ffi.sum(x, axis, keep))

def take(x, indices, axis=None):
    x = sym_utils.to_tensor(x)
    indices = sym_utils.to_tensor(indices)
    axis = sym_utils.to_any(axis)
    return Symbol.from_expr(ffi.take(x, indices, axis))

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
