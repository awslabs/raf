from ._ffi._tvm import _NodeBase
from ._ffi import _ndarray
from .base import register_mnm_node


@register_mnm_node("mnm.ndarray")
class ndarray(_NodeBase):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


def log(data):
    return _ndarray.log(data)


def exp(data):
    return _ndarray.exp(data)


def sqrt(data):
    return _ndarray.sqrt(data)


def zeros_like(data):
    return _ndarray.zeros_like(data)


def ones_like(data):
    return _ndarray.ones_like(data)


def sigmoid(data):
    return _ndarray.sigmoid(data)


def copy(data):
    return _ndarray.copy(data)


def floor(data):
    return _ndarray.floor(data)


def ceil(data):
    return _ndarray.ceil(data)


def trunc(data):
    return _ndarray.trunc(data)


def round(data):
    return _ndarray.round(data)


def sign(data):
    return _ndarray.sign(data)


def abs(data):
    return _ndarray.abs(data)


def tanh(data):
    return _ndarray.tanh(data)


def negative(data):
    return _ndarray.negative(data)


def logical_not(data):
    return _ndarray.logical_not(data)


def add(lhs, rhs):
    return _ndarray.add(lhs, rhs)


def subtract(lhs, rhs):
    return _ndarray.subtract(lhs, rhs)


def right_shift(lhs, rhs):
    return _ndarray.right_shift(lhs, rhs)


def left_shift(lhs, rhs):
    return _ndarray.left_shift(lhs, rhs)


def maximum(lhs, rhs):
    return _ndarray.maximum(lhs, rhs)


def minimum(lhs, rhs):
    return _ndarray.minimum(lhs, rhs)


def divide(lhs, rhs):
    return _ndarray.divide(lhs, rhs)


def multiply(lhs, rhs):
    return _ndarray.multiply(lhs, rhs)


def power(lhs, rhs):
    return _ndarray.power(lhs, rhs)


def mod(lhs, rhs):
    return _ndarray.mod(lhs, rhs)


def logical_and(lhs, rhs):
    return _ndarray.logical_and(lhs, rhs)


def logical_or(lhs, rhs):
    return _ndarray.logical_or(lhs, rhs)


def equal(lhs, rhs):
    return _ndarray.equal(lhs, rhs)


def not_equal(lhs, rhs):
    return _ndarray.not_equal(lhs, rhs)


def less(lhs, rhs):
    return _ndarray.less(lhs, rhs)


def less_equal(lhs, rhs):
    return _ndarray.less_equal(lhs, rhs)


def greater(lhs, rhs):
    return _ndarray.greater(lhs, rhs)


def greater_equal(lhs, rhs):
    return _ndarray.greater_equal(lhs, rhs)


def argmax(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.argmax(data, axis, keepdims, exclude)


def argmin(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.argmin(data, axis, keepdims, exclude)


def sum(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if axis and isinstance(axis, int) else axis
    return _ndarray.sum(data, axis, keepdims, exclude)


def max(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.max(data, axis, keepdims, exclude)


def min(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.min(data, axis, keepdims, exclude)


def prod(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.prod(data, axis, keepdims, exclude)


def mean(data, axis=None, keepdims=False, exclude=False):
    axis = [axis] if isinstance(axis, int) else axis
    return _ndarray.mean(data, axis, keepdims, exclude)


def conv2d(data,
           weight,
           strides=(1, 1),
           padding=(0, 0),
           dilation=(1, 1),
           groups=1,
           channels=None,
           kernel_size=None,
           data_layout="NCHW",
           kernel_layout="OIHW",
           out_layout="",
           out_dtype=""):
    return _ndarray.conv2d(data, weight, strides, padding, dilation,
                           groups, channels, kernel_size, data_layout,
                           kernel_layout, out_layout, out_dtype)


def conv2d_transpose(data,
                     weight,
                     strides=(1, 1),
                     padding=(0, 0),
                     dilation=(1, 1),
                     groups=1,
                     channels=None,
                     kernel_size=None,
                     data_layout="NCHW",
                     kernel_layout="OIHW",
                     output_padding=(0, 0),
                     out_dtype=""):
    return _ndarray.conv2d_transpose(data, weight, strides, padding, dilation,
                                     groups, channels, kernel_size, data_layout,
                                     kernel_layout, output_padding, out_dtype)


def deformable_conv2d(data,
                      offset,
                      weight,
                      strides=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      deformable_groups=1,
                      groups=1,
                      channels=None,
                      kernel_size=None,
                      data_layout='NCHW',
                      kernel_layout='OIHW',
                      out_layout='',
                      out_dtype=''):
    return _ndarray.deformable_conv2d(data, offset, weight, strides, padding, dilation,
                                      deformable_groups, groups, channels, kernel_size, data_layout,
                                      kernel_layout, out_layout, out_dtype)


def bias_add(data, bias, axis=1):
    return _ndarray.bias_add(data, bias, axis)


def dense(data, weight, units=None):
    return _ndarray.dense(data, weight, units)


def leaky_relu(data, alpha):
    return _ndarray.leaky_relu(data, alpha)


def prelu(data, alpha, axis=1):
    return _ndarray.prelu(data, alpha, axis)


def softmax(data, axis=-1):
    return _ndarray.softmax(data, axis)


def log_softmax(data, axis=-1):
    return _ndarray.log_softmax(data, axis)


def batch_flatten(data):
    return _ndarray.batch_flatten(data)


def relu(data):
    return _ndarray.relu(data)


def lrn(data, size=5, axis=1, bias=2, alpha=.00001, beta=0.75):
    return _ndarray.lrn(data, size, axis, alpha, beta, bias)


def l2_normalize(data, eps, axis=None):
    return _ndarray.l2_normalize(data, eps, axis)


def batch_norm(data,
               gamma,
               beta,
               moving_mean,
               moving_var,
               axis=1,
               epsilon=1e-5,
               center=True,
               scale=True):
    result = _ndarray.batch_norm(data,
                                 gamma,
                                 beta,
                                 moving_mean,
                                 moving_var,
                                 axis,
                                 epsilon,
                                 center,
                                 scale)
    return TupleWrapper(result, 3)


def batch_matmul(x, y):
    return _ndarray.batch_matmul(x, y)


def max_pool2d(data,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False):
    return _ndarray.max_pool2d(data, pool_size, strides, padding,
                               layout, ceil_mode)


def avg_pool2d(data,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False,
               count_include_pad=False):
    return _ndarray.avg_pool2d(data, pool_size, strides, padding,
                               layout, ceil_mode, count_include_pad)


def cast(data, dtype):
    return _ndarray.cast(data, dtype)


def expand_dims(data, axis, num_newaxis=1):
    return _ndarray.expand_dims(data, axis, num_newaxis)


def concatenate(data, axis):
    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    return _ndarray.concatenate(Tuple(data), axis)


def transpose(data, axes=None):
    if axes is not None:
        axes = list(axes)
    return _ndarray.transpose(data, axes)


def reshape(data, newshape):
    if isinstance(newshape, int):
        newshape = [newshape]
    return _ndarray.reshape(data, list(newshape))


def reshape_like(data, shape_like):
    return _ndarray.reshape_like(data, shape_like)


def take(data, indices, axis=None, mode="clip"):
    return _ndarray.take(data, indices, axis, mode)


def full(fill_value, shape=(), dtype=""):
    return _ndarray.full(fill_value, shape, dtype)


def zeros(shape, dtype="int32"):
    return _ndarray.zeros(shape, dtype)


def ones(shape, dtype):
    return _ndarray.ones(shape, dtype)


def full_like(data, fill_value):
    return _ndarray.full_like(data, fill_value)


def arange(start, stop=None, step=1, dtype="float32"):
    if stop is None:
        stop = start
        start = 0
    return _ndarray.arange(start, stop, step, dtype)


def repeat(data, repeats, axis):
    return _ndarray.repeat(data, repeats, axis)


def tile(data, reps):
    return _ndarray.tile(data, reps)


def reverse(data, axis):
    return _ndarray.reverse(data, axis)


def where(condition, x, y):
    return _ndarray.where(condition, x, y)


def squeeze(data, axis=None):
    return _ndarray.squeeze(data, axis)


def broadcast_to(data, shape):
    return _ndarray.broadcast_to(data, shape)


def gather_nd(data, indices):
    return _ndarray.gather_nd(data, indices)
