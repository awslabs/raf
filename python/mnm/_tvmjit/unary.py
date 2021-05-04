# pylint: disable=missing-function-docstring
"""Schedule registries for unary operators."""
import math
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg


_reg.register_broadcast_schedule("mnm.op.copy")
_reg.register_broadcast_schedule("mnm.op.ceil")
_reg.register_broadcast_schedule("mnm.op.floor")
_reg.register_broadcast_schedule("mnm.op.abs")
_reg.register_broadcast_schedule("mnm.op.erf")
_reg.register_broadcast_schedule("mnm.op.cos")
_reg.register_broadcast_schedule("mnm.op.sin")
_reg.register_broadcast_schedule("mnm.op.sign")
_reg.register_broadcast_schedule("mnm.op.round")
_reg.register_broadcast_schedule("mnm.op.exp")
_reg.register_broadcast_schedule("mnm.op.log")
_reg.register_broadcast_schedule("mnm.op.log2")
_reg.register_broadcast_schedule("mnm.op.sqrt")
_reg.register_broadcast_schedule("mnm.op.rsqrt")
_reg.register_broadcast_schedule("mnm.op.atan")
_reg.register_broadcast_schedule("mnm.op.relu")
_reg.register_broadcast_schedule("mnm.op.negative")
_reg.register_broadcast_schedule("mnm.op.sigmoid")
_reg.register_broadcast_schedule("mnm.op.tanh")

@register_compute("mnm.op.tanh_dx")
def tanh_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # grad = dy * (1 - tanh(x) * tanh(x)) = dy * (1 - y * y)
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape, lambda *idx: dy[idx] * (1 - y[idx] * y[idx]),
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_broadcast_schedule("mnm.op.tanh_dx")
_reg.register_broadcast_schedule("mnm.op.trunc")

@register_compute("mnm.op.erf_dx")
def erf_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape,
                            lambda *idx: _tvm.tir.const(2 / math.sqrt(math.pi), dtype=dy.dtype)
                            * _tvm.te.exp(-x[idx] * x[idx]) * dy[idx],
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_broadcast_schedule("mnm.op.erf_dx")
_reg.register_injective_schedule("mnm.op.zeros_like")
_reg.register_injective_schedule("mnm.op.ones_like")

@register_compute("mnm.op.sqrt_dx")
def sqrt_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, y, dy = inputs
    return [_tvm.te.compute(x.shape,
                            lambda *idx: dy[idx] / (y[idx] + y[idx]),
                            tag=_tvm.topi.tag.ELEMWISE)]

_reg.register_injective_schedule("mnm.op.sqrt_dx")
