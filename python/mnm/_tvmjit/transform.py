# pylint: disable=missing-function-docstring
"""Compute definition and schedules for data transform operators"""
from .._lib import register_compute
from .._lib import topi as _topi
from .._lib import tvm as _tvm  # pylint: disable=unused-import
from .._lib import _reg


@register_compute("mnm.op.transpose_dx")
def transpose_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    dy = inputs[2]
    axes = list(_topi.util.get_const_tuple(attrs.axes))
    axes_inverse = axes.copy()
    for idx, i in enumerate(axes):
        axes_inverse[i] = idx
    out = _topi.transpose(dy, axes=tuple(axes_inverse))
    return [out]


_reg.register_injective_schedule("mnm.op.transpose_dx")
_reg.register_injective_schedule("mnm.op.transpose")
_reg.register_injective_schedule("mnm.op.split")
_reg.register_injective_schedule("mnm.op.take")
_reg.register_injective_schedule("mnm.op.sequence_mask")
_reg.register_injective_schedule("mnm.op.reverse_sequence")
_reg.register_injective_schedule("mnm.op.concatenate")
_reg.register_injective_schedule("mnm.op.reverse")
_reg.register_injective_schedule("mnm.op.stack")

_reg.register_broadcast_schedule("mnm.op.broadcast_to")
_reg.register_broadcast_schedule("mnm.op.broadcast_to_like")
_reg.register_broadcast_schedule("mnm.op.clip")
_reg.register_broadcast_schedule("mnm.op.repeat")


@register_compute("mnm.op.clip_dx")
def clip_dx_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    x = inputs[0]
    grad = inputs[1]
    a_min = attrs.a_min
    a_max = attrs.a_max

    def _select(*indices):
        return _tvm.tir.if_then_else(_tvm.tir.any(x[indices] <= a_min,
                                                  x[indices] >= a_max),
                                     0, grad(*indices))
    return [_tvm.te.compute(x.shape, _select)]


_reg.register_injective_schedule("mnm.op.clip_dx")
