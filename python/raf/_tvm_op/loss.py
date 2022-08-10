# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring
"""Compute definition and schedules for loss functions."""
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member


@register_compute("raf.op.tvm.smooth_l1_loss")
def smooth_l1_loss_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    delta = _topi.abs(_topi.subtract(true, pred))
    mul = 1
    for i in pred.shape:
        mul *= i

    temp = _tvm.te.compute(
        pred.shape,
        lambda *ind: _tvm.tir.if_then_else(
            delta[ind] >= 1, delta[ind] - _tvm.tir.const(0.5), 0.5 * delta[ind] * delta[ind]
        ),
    )
    loss = _topi.transform.reshape(
        _topi.sum(temp / mul, axis=tuple(range(0, len(pred.shape)))), [1]
    )
    return [loss]


_reg.register_injective_schedule("raf.op.tvm.smooth_l1_loss")


@register_compute("raf.op.tvm.smooth_l1_loss_dpred")
def smooth_l1_loss_dpred_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    delta = _topi.abs(_topi.subtract(true, pred))
    mul = 1
    for i in pred.shape:
        mul *= i

    dpred = _tvm.te.compute(
        pred.shape,
        lambda *ind: _tvm.tir.if_then_else(
            delta[ind] >= 1,
            _tvm.tir.if_then_else(pred[ind] > true[ind], _tvm.tir.const(-1), _tvm.tir.const(1)),
            true[ind] - pred[ind],
        ),
    )

    return [dpred / mul]


_reg.register_broadcast_schedule("raf.op.tvm.smooth_l1_loss_dpred")


@register_compute("raf.op.tvm.smooth_l1_loss_dtrue")
def smooth_l1_loss_dtrue_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    delta = _topi.abs(_topi.subtract(true, pred))
    mul = 1
    for i in pred.shape:
        mul *= i

    dtrue = _tvm.te.compute(
        pred.shape,
        lambda *ind: _tvm.tir.if_then_else(
            delta[ind] >= 1,
            _tvm.tir.if_then_else(pred[ind] > true[ind], _tvm.tir.const(1), _tvm.tir.const(-1)),
            -true[ind] + pred[ind],
        ),
    )

    return [dtrue / mul]


_reg.register_broadcast_schedule("raf.op.tvm.smooth_l1_loss_dtrue")


@register_compute("raf.op.tvm.nll_loss")
def nll_loss_compute(attrs, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    n, c = pred.shape
    dtype = pred.dtype
    if dtype == "float16":
        pred = pred.astype("float32")

    if true.ndim == 1:  # one-host label encoding

        def fcompute_one_hot(i):  # pylint: disable=unused-argument
            return -pred[i, true[i]] / n

        loss = _tvm.te.compute((n,), fcompute_one_hot, tag=_tvm.topi.tag.INJECTIVE)
        loss = _topi.sum(loss, axis=[0], keepdims=True)
    else:  # sparse label encoding

        def fcompute_sparse(x):  # pylint: disable=unused-argument
            redn = _tvm.te.reduce_axis((0, n), name="rn")
            redc = _tvm.te.reduce_axis((0, c), name="rc")
            return _tvm.te.sum(-pred[redn, redc] * true[redn, redc] / n, axis=[redc, redn])

        loss = _tvm.te.compute((1,), fcompute_sparse, tag=_tvm.topi.tag.COMM_REDUCE)

    return [loss.astype(dtype)]


_reg.register_reduce_schedule("raf.op.tvm.nll_loss")


@register_compute("raf.op.tvm.nll_loss_dpred")
def nllloss_dpred_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    dy, true, pred = inputs
    n, c = pred.shape
    dy = dy if dy.ndim == 0 else dy[0]

    if true.ndim == 1:  # one-hot label encoding
        dpred = _tvm.te.compute(
            (n, c), lambda x, y: _tvm.tir.if_then_else(y == true[x], -dy / n, _tvm.tir.const(0))
        )
    else:  # sparse label encoding
        dpred = _tvm.te.compute((n, c), lambda x, y: -true[x, y] / n) * dy
    return [dpred]


_reg.register_broadcast_schedule("raf.op.tvm.nll_loss_dpred")


@register_compute("raf.op.tvm.nll_loss_dtrue")
def nllloss_dtrue_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    dy, true, pred = inputs
    n, c = pred.shape

    dtrue = _tvm.te.compute((n, c), lambda x, y: -pred[x, y] / n) * dy

    if true.ndim == 1:  # for one_hot label encoding
        dtrue = _topi.sum(dtrue, axis=[1], keepdims=False)

    return [dtrue]


_reg.register_broadcast_schedule("raf.op.tvm.nll_loss_dtrue")


@register_compute("raf.op.tvm.cross_entropy")
def cross_entropy_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    true, pred = inputs
    log_pred = _topi.nn.log_softmax(pred)
    return nll_loss_compute(attr, [true, log_pred], output_type)


_reg.register_broadcast_schedule("raf.op.tvm.cross_entropy")


@register_compute("raf.op.tvm.cross_entropy_dpred")
def cross_entropy_dpred_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    # The grad function of cross_entropy decomposes cross_entropy_dpred to a series of RAF IR ops
    # so this function is not used. It only kept in case we want to have a powerful schedule
    # especially for this op in the future.
    _, _, pred = inputs
    log_pred = _topi.nn.log_softmax(pred)
    dpred = nllloss_dpred_compute(attr, inputs, output_type)[0]
    return [dpred - _topi.exp(log_pred) * _topi.sum(dpred, -1, True)]


_reg.register_broadcast_schedule("raf.op.tvm.cross_entropy_dpred")


@register_compute("raf.op.tvm.cross_entropy_dtrue")
def cross_entropy_dtrue_compute(attr, inputs, output_type):  # pylint: disable=unused-argument
    return nllloss_dtrue_compute(attr, inputs, output_type)


_reg.register_broadcast_schedule("raf.op.tvm.cross_entropy_dtrue")
