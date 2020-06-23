from .._lib import OpPattern, register_compute
from .._lib import topi as _topi  # pylint: disable=unused-import
from .._lib import tvm as _tvm
from .._lib import _reg


@register_compute("mnm.op.nll_loss")
def nll_loss_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    n, c = pred.shape
    redn = _tvm.te.reduce_axis((0, n), name='rn')
    redc = _tvm.te.reduce_axis((0, c), name='rc')

    def fcompute(x):  # pylint: disable=unused-argument
        return _tvm.te.sum(-pred[redn, redc] * true[redn, redc] / n,
                           axis=[redc, redn])

    loss = _tvm.te.compute((1, ), fcompute)
    return [loss]


_reg.register_injective_schedule("mnm.op.nll_loss")
_reg.register_pattern("mnm.op.nll_loss", OpPattern.INJECTIVE)


@register_compute("mnm.op.nll_loss_dpred")
def nllloss_dpred_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    true, _ = inputs
    n, c = true.shape
    return [_tvm.te.compute((n, c), lambda x, y: -true[x, y] / n)]


_reg.register_broadcast_schedule("mnm.op.nll_loss_dpred")
_reg.register_pattern("mnm.op.nll_loss_dpred", OpPattern.ELEMWISE)


@register_compute("mnm.op.nll_loss_dtrue")
def nllloss_dtrue_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    _, pred = inputs
    n, c = pred.shape
    return [_tvm.te.compute((n, c), lambda x, y: -pred[x, y] / n)]

_reg.register_broadcast_schedule("mnm.op.nll_loss_dtrue")
_reg.register_pattern("mnm.op.nll_loss_dtrue", OpPattern.ELEMWISE)