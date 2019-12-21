from .._lib import tvm as _tvm
from .._lib import topi as _topi
from .._lib import relay as _relay

@_relay.op.register_compute("mnm.op.sgd")
def compute(attr, inputs, output_type, target): # pylint: disable=unused-argument
    learning_rate, mu = attr.learning_rate, attr.mu
    x0, dx, v0 = inputs
    def fcomputev(*args):
        return mu * v0(*args) + dx(*args)
    v1 = _tvm.compute(v0.shape, fcomputev)
    def fcomputex(*args):
        return x0(*args) - learning_rate * v1(*args)
    x1 = _tvm.compute(x0.shape, fcomputex)
    return [v1, x1]

@_relay.op.register_schedule("mnm.op.sgd")
def schedule(attr, outputs, target): # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_elemwise(outputs)

_relay.op.register_pattern("mnm.op.sgd", _relay.op.OpPattern.ELEMWISE)
