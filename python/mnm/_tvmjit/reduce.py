# pylint: disable=missing-function-docstring
"""Reduction compute definition and schedules."""
from .._lib import register_compute
from .._lib import topi as _topi
from .._lib import tvm as _tvm
from .._lib import _reg


@register_compute("mnm.op.sum")
def sum_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    x = inputs[0]
    axes = [int(i) for i in attrs.axis]
    keep = [int(i) for i in attrs.keep]
    if not axes:
        # TODO(@were): It seems that TVM create view may crash, I cannot directly return [x]
        return [_tvm.te.compute(x.shape, lambda *args: x(*args))] # pylint: disable=unnecessary-lambda
    if len(keep) == 1:
        keep = [keep[0]] * len(axes)
    # Fallback to TOPI
    if keep == [keep[0]] * len(axes):
        return [_topi.sum(x, axes, keep[0])]
    axes = sorted(zip(axes, keep))
    red_axis = [_tvm.te.reduce_axis((0, x.shape[i])) for i, _ in axes]
    shape = list(x.shape)
    for i, j in axes:
        shape[i] = 1 if j else None
    shape = [i for i in shape if i is not None]

    def fcompute(*args):
        scan = list(args[::-1])
        reds = red_axis[::-1]
        idx = []
        for i in range(len(x.shape)):
            if (i, True) in axes:
                idx.append(reds.pop())
                scan.pop()
            elif (i, False) in axes:
                idx.append(reds.pop())
            else:
                idx.append(scan.pop())
        return _tvm.te.sum(x(*idx), axis=red_axis)

    return [_tvm.te.compute(shape, fcompute)]


_reg.register_injective_schedule("mnm.op.sum")

_reg.register_reduce_schedule("mnm.op.argmax")
_reg.register_reduce_schedule("mnm.op.argmin")
_reg.register_reduce_schedule("mnm.op.max")
_reg.register_reduce_schedule("mnm.op.min")
_reg.register_reduce_schedule("mnm.op.all")
_reg.register_reduce_schedule("mnm.op.any")
_reg.register_reduce_schedule("mnm.op.mean")


def axis_reverse(input_axis):
    # get the reverse axis to change axis back
    # e.g.: input_axis = [1, 2, 0, 3]
    #       the reverse_axis = [2, 0, 1, 3]
    reverse_axis = input_axis.copy()
    for i, axis in enumerate(input_axis):
        reverse_axis[axis] = i
    return reverse_axis


def mul_shapes(shape_list, axis_list):
    # get the product of shapes in given axis
    out = 1
    for axis in axis_list:
        out *= shape_list[axis]
    return out


@register_compute("mnm.op.mean_dx")
def mean_dx_compute(attrs, inputs, output_type): # pylint: disable=unused-argument
    x = inputs[0]
    dy = inputs[2]
    axis = list(_topi.util.get_const_tuple(attrs.axis))
    keepdims = attrs.keepdims
    shape_mul = mul_shapes(x.shape, axis)
    def _elem_div(*indices):
        return dy[indices] / shape_mul
    out = _tvm.te.compute(dy.shape, _elem_div)
    # if keepdims = True, repeat the elements in those reduced axis
    if keepdims:
        for repeat_axis in axis:
            out = _topi.repeat(out, int(x.shape[repeat_axis]), repeat_axis)
    # if keepdims = False, using broadcast_to to expand the dimension
    else:
        left_axis = list(set(range(len(x.shape))) - set(axis))
        expand_axis = axis + left_axis
        reverse_axis = axis_reverse(expand_axis)
        out = _topi.broadcast_to(out, _topi.transpose(x, axes=expand_axis).shape)
        out = _topi.transpose(out, axes=reverse_axis)
    return [out]


_reg.register_injective_schedule("mnm.op.mean_dx")
