import mnm._ffi.op.sym as ffi
from mnm._core.ndarray import Symbol
from . import sym_utils

# pylint: disable=invalid-name,line-too-long,too-many-arguments
__all__ = [
    "add", "avg_pool2d", "avg_pool2d_dx", "batch_flatten", "batch_norm_infer",
    "batch_norm_train", "batch_norm_train_dxwb", "bias_add", "collapse_sum_like", "conv2d",
    "conv2d_dw", "conv2d_dx", "divide", "equal", "greater",
    "greater_equal", "less", "less_equal", "log_softmax", "log_softmax_dx",
    "logical_not", "matmul", "max_pool2d", "max_pool2d_dx", "mod",
    "multiply", "negative", "nll_loss", "nll_loss_dpred", "nll_loss_dtrue",
    "not_equal", "relu", "relu_dx", "reshape_like", "sigmoid",
    "sigmoid_dx", "softmax", "softmax_dx", "subtract", "tanh",
    "tanh_dx",
]

def add(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.add(x1, x2, out, where))
def avg_pool2d(x, kernel, stride=None, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_optional_int_tuple(stride)
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
def batch_norm_infer(x, running_mean, running_var, w=None, b=None, eps=1e-05, momentum=0.1):
    x = sym_utils.to_tensor(x)
    running_mean = sym_utils.to_tensor(running_mean)
    running_var = sym_utils.to_tensor(running_var)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    eps = sym_utils.to_double(eps)
    momentum = sym_utils.to_double(momentum)
    return Symbol.from_expr(ffi.batch_norm_infer(x, running_mean, running_var, w, b, eps, momentum))
def batch_norm_train(x, running_mean, running_var, w=None, b=None, eps=1e-05, momentum=0.1):
    x = sym_utils.to_tensor(x)
    running_mean = sym_utils.to_tensor(running_mean)
    running_var = sym_utils.to_tensor(running_var)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    eps = sym_utils.to_double(eps)
    momentum = sym_utils.to_double(momentum)
    return Symbol.from_expr(ffi.batch_norm_train(x, running_mean, running_var, w, b, eps, momentum))
def batch_norm_train_dxwb(dy, x, w, b):
    dy = sym_utils.to_tensor(dy)
    x = sym_utils.to_tensor(x)
    w = sym_utils.to_tensor(w)
    b = sym_utils.to_tensor(b)
    return Symbol.from_expr(ffi.batch_norm_train_dxwb(dy, x, w, b))
def bias_add(x, b, axis):
    x = sym_utils.to_tensor(x)
    b = sym_utils.to_tensor(b)
    axis = sym_utils.to_int(axis)
    return Symbol.from_expr(ffi.bias_add(x, b, axis))
def collapse_sum_like(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.collapse_sum_like(x, shape))
def conv2d(x, w, stride=1, padding=0, dilation=1, groups=1):
    x = sym_utils.to_tensor(x)
    w = sym_utils.to_tensor(w)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d(x, w, stride, padding, dilation, groups))
def conv2d_dw(x_or_w, y, dy, stride, padding, dilation, groups):
    x_or_w = sym_utils.to_tensor(x_or_w)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d_dw(x_or_w, y, dy, stride, padding, dilation, groups))
def conv2d_dx(x_or_w, y, dy, stride, padding, dilation, groups):
    x_or_w = sym_utils.to_tensor(x_or_w)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    stride = sym_utils.to_int_tuple(stride)
    padding = sym_utils.to_int_tuple(padding)
    dilation = sym_utils.to_int_tuple(dilation)
    groups = sym_utils.to_int(groups)
    return Symbol.from_expr(ffi.conv2d_dx(x_or_w, y, dy, stride, padding, dilation, groups))
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
def matmul(a, b, transpose_a=False, transpose_b=False):
    a = sym_utils.to_tensor(a)
    b = sym_utils.to_tensor(b)
    transpose_a = sym_utils.to_bool(transpose_a)
    transpose_b = sym_utils.to_bool(transpose_b)
    return Symbol.from_expr(ffi.matmul(a, b, transpose_a, transpose_b))
def max_pool2d(x, kernel, stride=None, padding=0, dilation=1, ceil_mode=False, include_pad=True):
    x = sym_utils.to_tensor(x)
    kernel = sym_utils.to_int_tuple(kernel)
    stride = sym_utils.to_optional_int_tuple(stride)
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
def nll_loss_dpred(loss, y_true, y_pred):
    loss = sym_utils.to_tensor(loss)
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dpred(loss, y_true, y_pred))
def nll_loss_dtrue(loss, y_true, y_pred):
    loss = sym_utils.to_tensor(loss)
    y_true = sym_utils.to_tensor(y_true)
    y_pred = sym_utils.to_tensor(y_pred)
    return Symbol.from_expr(ffi.nll_loss_dtrue(loss, y_true, y_pred))
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
def reshape_like(x, shape):
    x = sym_utils.to_tensor(x)
    shape = sym_utils.to_int_tuple(shape)
    return Symbol.from_expr(ffi.reshape_like(x, shape))
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
def subtract(x1, x2, out=None, where=None):
    x1 = sym_utils.to_any(x1)
    x2 = sym_utils.to_any(x2)
    out = sym_utils.to_any(out)
    where = sym_utils.to_any(where)
    return Symbol.from_expr(ffi.subtract(x1, x2, out, where))
def tanh(x):
    x = sym_utils.to_any(x)
    return Symbol.from_expr(ffi.tanh(x))
def tanh_dx(x, y, dy):
    x = sym_utils.to_any(x)
    y = sym_utils.to_tensor(y)
    dy = sym_utils.to_tensor(dy)
    return Symbol.from_expr(ffi.tanh_dx(x, y, dy))
