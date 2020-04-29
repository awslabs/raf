from .._lib import (OpPattern, register_compute, register_pattern,
                    register_schedule)
from .._lib import topi as _topi
from .._lib import tvm as _tvm  # pylint: disable=unused-import


@register_compute("mnm.op.transpose_dx")
def compute(attrs, inputs, output_type, target):  # pylint: disable=unused-argument
    dy = inputs[2]
    axes = list(_topi.util.get_const_tuple(attrs.axes))
    axes_inverse = axes.copy()
    for idx, i in enumerate(axes):
        axes_inverse[i] = idx
    out = _topi.transpose(dy, axes=tuple(axes_inverse))
    return [out]


@register_schedule("mnm.op.transpose_dx")
def schedule(attr, outputs, target):  # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_injective(outputs)


register_pattern("mnm.op.transpose_dx", OpPattern.INJECTIVE)
