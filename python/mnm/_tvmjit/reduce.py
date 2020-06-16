from .._lib import OpPattern, register_compute
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
_reg.register_pattern("mnm.op.sum", OpPattern.COMM_REDUCE)

_reg.register_reduce_schedule("mnm.op.argmax")
_reg.register_reduce_schedule("mnm.op.argmin")
_reg.register_reduce_schedule("mnm.op.all")
_reg.register_reduce_schedule("mnm.op.any")
