# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
from .._lib import register_compute
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
