from .._lib import register_compute
from .._lib import topi as _topi  # pylint: disable=unused-import
from .._lib import tvm as _tvm
from .._lib import _reg


@register_compute("mnm.op.sgd")
def sgd_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    learning_rate, mu = attr.learning_rate, attr.mu  # pylint: disable=invalid-name
    x0, dx, v0 = inputs

    def fcomputev(*args):
        return mu * v0(*args) + dx(*args)

    v1 = _tvm.te.compute(v0.shape, fcomputev)

    def fcomputex(*args):
        return x0(*args) - learning_rate * v1(*args)

    x1 = _tvm.te.compute(x0.shape, fcomputex)
    return [v1, x1]

_reg.register_broadcast_schedule("mnm.op.sgd")
