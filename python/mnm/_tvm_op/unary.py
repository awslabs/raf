# pylint: disable=missing-function-docstring
"""Schedule registries for unary operators."""
import math
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg


_reg.register_broadcast_schedule("mnm.op.tvm.copy")
_reg.register_broadcast_schedule("mnm.op.tvm.ceil")
_reg.register_broadcast_schedule("mnm.op.tvm.floor")
_reg.register_broadcast_schedule("mnm.op.tvm.abs")
_reg.register_broadcast_schedule("mnm.op.tvm.erf")
_reg.register_broadcast_schedule("mnm.op.tvm.cos")
_reg.register_broadcast_schedule("mnm.op.tvm.sin")
_reg.register_broadcast_schedule("mnm.op.tvm.sign")
_reg.register_broadcast_schedule("mnm.op.tvm.round")
_reg.register_broadcast_schedule("mnm.op.tvm.exp")
_reg.register_broadcast_schedule("mnm.op.tvm.log")
_reg.register_broadcast_schedule("mnm.op.tvm.log2")
_reg.register_broadcast_schedule("mnm.op.tvm.sqrt")
_reg.register_broadcast_schedule("mnm.op.tvm.rsqrt")
_reg.register_broadcast_schedule("mnm.op.tvm.atan")
_reg.register_broadcast_schedule("mnm.op.tvm.relu")
_reg.register_broadcast_schedule("mnm.op.tvm.negative")
_reg.register_broadcast_schedule("mnm.op.tvm.sigmoid")
_reg.register_broadcast_schedule("mnm.op.tvm.tanh")

@register_compute("mnm.op.tvm.tanh_dx")
def tanh_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # grad = dy * (1 - tanh(x) * tanh(x)) = dy * (1 - y * y)
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape, lambda *idx: dy[idx] * (1 - y[idx] * y[idx]),
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_broadcast_schedule("mnm.op.tvm.tanh_dx")
_reg.register_broadcast_schedule("mnm.op.tvm.trunc")

@register_compute("mnm.op.tvm.erf_dx")
def erf_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape,
                            lambda *idx: _tvm.tir.const(2 / math.sqrt(math.pi), dtype=dy.dtype)
                            * _tvm.te.exp(-x[idx] * x[idx]) * dy[idx],
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_broadcast_schedule("mnm.op.tvm.erf_dx")
_reg.register_injective_schedule("mnm.op.tvm.zeros_like")
_reg.register_injective_schedule("mnm.op.tvm.ones_like")

@register_compute("mnm.op.tvm.sqrt_dx")
def sqrt_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape,
                            lambda *idx: dy[idx] / (y[idx] + y[idx]),
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_injective_schedule("mnm.op.tvm.sqrt_dx")

@register_compute("mnm.op.tvm.gelu")
def gelu_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    data = inputs[0]
    # gelu is data  * normcdf(data)
    const_point_5 = _tvm.tir.const(0.5, dtype=data.dtype)
    const_1 = _tvm.tir.const(1, dtype=data.dtype)
    const_sqrt_2 = _tvm.tir.const(math.sqrt(2), dtype=data.dtype)
    return [data * (const_point_5 * (const_1 + _tvm.topi.erf(data / const_sqrt_2)))]

_reg.register_injective_schedule("mnm.op.tvm.gelu")

@register_compute("mnm.op.tvm.gelu_dx")
def gelu_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs
    const_point_5 = _tvm.tir.const(0.5, dtype=x.dtype)
    const_minus_point_5 = _tvm.tir.const(-0.5, dtype=x.dtype)
    const_1 = _tvm.tir.const(1, dtype=x.dtype)
    const_sqrt_2 = _tvm.tir.const(math.sqrt(2), dtype=x.dtype)
    const_sqrt_pi = _tvm.tir.const(math.sqrt(math.pi), dtype=x.dtype)
    # cdf = 0.5 * (1 + erf(x/sqrt(2)))
    cdf = const_point_5 * (const_1 + _tvm.topi.erf(x / const_sqrt_2))
    # beta = 1 / sqrt(2*pi)
    const_beta = const_1 / (const_sqrt_2 * const_sqrt_pi)
    # pdf = beta * e^(-0.5x^2)
    pdf = const_beta * _tvm.topi.exp(x * x * const_minus_point_5)
    return [dy * (cdf + x * pdf)]

_reg.register_injective_schedule("mnm.op.tvm.gelu_dx")
