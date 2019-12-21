from .._lib import tvm as _tvm
from .._lib import topi as _topi
from .._lib import relay as _relay


@_relay.op.register_compute("mnm.op.add_dx")
def add_dx_compute(attr, inputs, output_type, target): # pylint: disable=unused-argument
    # pylint: disable=too-many-locals
    a, _, _, dy = inputs
    n_a = len(a.shape)
    n_dy = len(dy.shape)
    n = max(n_a, n_dy)
    # TODO(@were): Support Any shape.
    shapea = [1] * (n - n_a) + [int(i.value) for i in a.shape]
    shapedy = [1] * (n - n_dy) + [int(i.value) for i in dy.shape]
    idx = []
    for i, j in zip(shapea, shapedy):
        if i == j:
            idx.append(None)
        elif i == 1:
            idx.append(_tvm.reduce_axis((0, j)))
        else:
            assert False, f'{shapea} and {shapedy} does not match'
    def fcompute(*args):
        axis = []
        offset = n - n_a
        for i in range(offset):
            assert idx[i] is not None
            axis.append(idx[i])
        for i in range(offset, n):
            if idx[i] is None:
                idx[i] = args[i - offset]
            else:
                axis.append(idx[i])
        return _tvm.sum(dy(*idx), axis=axis) if axis else dy(*idx)
    return [_tvm.compute(a.shape, fcompute)]


@_relay.op.register_schedule("mnm.op.add_dx")
def add_dx_schedule(attr, outputs, target): # pylint: disable=unused-argument
    return _topi.generic.schedule_injective(outputs)

_relay.op.register_pattern("mnm.op.add_dx", _relay.op.OpPattern.INJECTIVE)
