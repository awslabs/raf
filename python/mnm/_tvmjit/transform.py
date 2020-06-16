from .._lib import OpPattern, register_compute
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
_reg.register_pattern("mnm.op.transpose_dx", OpPattern.INJECTIVE)

_reg.register_injective_schedule("mnm.op.transpose")
_reg.register_injective_schedule("mnm.op.split")
_reg.register_injective_schedule("mnm.op.take")
_reg.register_injective_schedule("mnm.op.sequence_mask")
_reg.register_injective_schedule("mnm.op.concatenate")

_reg.register_broadcast_schedule("mnm.op.broadcast_to")
_reg.register_broadcast_schedule("mnm.op.broadcast_to_like")
