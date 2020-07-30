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
_reg.register_broadcast_schedule("mnm.op.exp")
_reg.register_broadcast_schedule("mnm.op.log")
_reg.register_broadcast_schedule("mnm.op.sqrt")
_reg.register_broadcast_schedule("mnm.op.atan")
_reg.register_broadcast_schedule("mnm.op.relu")

@register_compute("mnm.op.erf_dx")
def erf_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    x, dy, y = inputs
    return [_tvm.te.compute(x.shape,
                            lambda *idx: _tvm.tir.const(2 / math.sqrt(math.pi), dtype=dy.dtype)
                            * _tvm.te.exp(-x[idx] * x[idx]) * dy[idx])]

_reg.register_broadcast_schedule("mnm.op.erf_dx")
