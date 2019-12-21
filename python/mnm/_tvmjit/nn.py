from .._lib import tvm as _tvm
from .._lib import topi as _topi
from .._lib import relay as _relay


@_relay.op.register_compute("mnm.op.nll_loss")
def nll_loss_compute(attrs, inputs, output_type, target): # pylint: disable=unused-argument
    true, pred = inputs
    n, c = pred.shape
    redn = _tvm.reduce_axis((0, n), name='rn')
    redc = _tvm.reduce_axis((0, c), name='rc')
    def fcompute(x):  # pylint: disable=unused-argument
        return _tvm.sum(-pred[redn, redc] * true[redn, redc] / n, axis=[redc, redn])
    loss = _tvm.compute((1, ), fcompute)
    return [loss]


@_relay.op.register_schedule("mnm.op.nll_loss")
def nll_loss_schedule(attr, outputs, target): # pylint: disable=unused-argument
    return _topi.generic.schedule_injective(outputs)


_relay.op.register_pattern("mnm.op.nll_loss", _relay.op.OpPattern.INJECTIVE)


@_relay.op.register_compute("mnm.op.nll_loss_dpred")
def nllloss_dpred_compute(attr, inputs, output_type, target): # pylint: disable=unused-argument
    true, _, _ = inputs
    n, c = true.shape
    return [_tvm.compute((n, c), lambda x, y: -true[x, y] / n)]


@_relay.op.register_schedule("mnm.op.nll_loss_dpred")
def nllloss_dpred_schedule(attr, outputs, target): # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_elemwise(outputs)


_relay.op.register_pattern("mnm.op.nll_loss_dpred", _relay.op.OpPattern.ELEMWISE)


@_relay.op.register_compute("mnm.op.nll_loss_dtrue")
def nllloss_dtrue_compute(attr, inputs, output_type, target): # pylint: disable=unused-argument
    _, pred, _ = inputs
    n, c = pred.shape
    return [_tvm.compute((n, c), lambda x, y: -pred[x, y] / n)]


@_relay.op.register_schedule("mnm.op.nll_loss_dtrue")
def nllloss_dtrue_schedule(attr, outputs, target): # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_elemwise(outputs)


_relay.op.register_pattern("mnm.op.nll_loss_dtrue", _relay.op.OpPattern.ELEMWISE)


@_relay.op.register_compute("mnm.op.bias_add_db")
def bias_add_db(attr, inputs, output_type, target):  # pylint: disable=unused-argument
    dy, b = inputs
    axis = attr.axis
    reds = [_tvm.reduce_axis((0, j)) if i != axis else None for i, j in enumerate(dy.shape)]
    def fcompute(x):
        reds[axis] = x
        return _tvm.sum(dy(*reds), axis=reds[:axis] + reds[axis+1:])
    db = _tvm.compute(b.shape, fcompute)
    return [db]

@_relay.op.register_schedule("mnm.op.bias_add_db")
def nllloss_back_schedule(attr, outputs, target): # pylint: disable=unused-argument
    with target:
        return _topi.generic.schedule_injective(outputs)

_relay.op.register_pattern("mnm.op.bias_add_db", _relay.op.OpPattern.INJECTIVE)
