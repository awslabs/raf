# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring, unused-argument
"""Schedule registries for unary operators."""
import math
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg


_reg.register_broadcast_schedule("raf.op.tvm.copy")
_reg.register_broadcast_schedule("raf.op.tvm.ceil")
_reg.register_broadcast_schedule("raf.op.tvm.floor")
_reg.register_broadcast_schedule("raf.op.tvm.abs")
_reg.register_broadcast_schedule("raf.op.tvm.erf")
_reg.register_broadcast_schedule("raf.op.tvm.cos")
_reg.register_broadcast_schedule("raf.op.tvm.sin")
_reg.register_broadcast_schedule("raf.op.tvm.sign")
_reg.register_broadcast_schedule("raf.op.tvm.round")
_reg.register_broadcast_schedule("raf.op.tvm.exp")
_reg.register_broadcast_schedule("raf.op.tvm.log")
_reg.register_broadcast_schedule("raf.op.tvm.log2")
_reg.register_broadcast_schedule("raf.op.tvm.sqrt")
_reg.register_broadcast_schedule("raf.op.tvm.rsqrt")
_reg.register_broadcast_schedule("raf.op.tvm.atan")
_reg.register_broadcast_schedule("raf.op.tvm.relu")
_reg.register_broadcast_schedule("raf.op.tvm.negative")
_reg.register_broadcast_schedule("raf.op.tvm.sigmoid")
_reg.register_broadcast_schedule("raf.op.tvm.tanh")


def select_unary_dx_input(attrs, inputs, x_or_y):
    """Select the required input based on the grad_mode to calculate the gradient.
    x_or_y=True selects x; otherwise y.

    if grad_mode == "both":
        x, y, dy = inputs
    else:
        x_or_y, dy = inputs
    """
    grad_mode = attrs.grad_mode
    x_or_y_idx = 1 if not x_or_y and grad_mode == "both" else 0
    return inputs[x_or_y_idx], inputs[-1]


@register_compute("raf.op.tvm.tanh_dx")
def tanh_dx_compute(attrs, inputs, output_type):
    # grad = dy * (1 - tanh(x) * tanh(x)) = dy * (1 - y * y)
    y, dy = select_unary_dx_input(attrs, inputs, False)
    return [
        _tvm.te.compute(
            y.shape, lambda *idx: dy[idx] * (1 - y[idx] * y[idx]), tag=_tvm.topi.tag.ELEMWISE
        )
    ]


_reg.register_broadcast_schedule("raf.op.tvm.tanh_dx")
_reg.register_broadcast_schedule("raf.op.tvm.trunc")


@register_compute("raf.op.tvm.erf_dx")
def erf_dx_compute(attrs, inputs, output_type):
    x, dy = select_unary_dx_input(attrs, inputs, True)
    return [
        _tvm.te.compute(
            x.shape,
            lambda *idx: _tvm.tir.const(2 / math.sqrt(math.pi), dtype=dy.dtype)
            * _tvm.te.exp(-x[idx] * x[idx])
            * dy[idx],
            tag=_tvm.topi.tag.ELEMWISE,
        )
    ]


_reg.register_broadcast_schedule("raf.op.tvm.erf_dx")
_reg.register_injective_schedule("raf.op.tvm.zeros_like")
_reg.register_injective_schedule("raf.op.tvm.ones_like")


@register_compute("raf.op.tvm.sqrt_dx")
def sqrt_dx_compute(attrs, inputs, output_type):
    y, dy = select_unary_dx_input(attrs, inputs, False)
    return [
        _tvm.te.compute(
            y.shape, lambda *idx: dy[idx] / (y[idx] + y[idx]), tag=_tvm.topi.tag.ELEMWISE
        )
    ]


_reg.register_injective_schedule("raf.op.tvm.sqrt_dx")


@register_compute("raf.op.tvm.gelu")
def gelu_compute(attrs, inputs, output_type):
    data = inputs[0]
    # gelu is data  * normcdf(data)
    const_point_5 = _tvm.tir.const(0.5, dtype=data.dtype)
    const_1 = _tvm.tir.const(1, dtype=data.dtype)
    const_sqrt_2 = _tvm.tir.const(math.sqrt(2), dtype=data.dtype)
    return [data * (const_point_5 * (const_1 + _tvm.topi.erf(data / const_sqrt_2)))]


_reg.register_injective_schedule("raf.op.tvm.gelu")


@register_compute("raf.op.tvm.gelu_dx")
def gelu_dx_compute(attrs, inputs, output_type):
    x, dy = select_unary_dx_input(attrs, inputs, True)
    const_point_5 = _tvm.tir.const(0.5, dtype=x.dtype)
    const_minus_point_5 = _tvm.tir.const(-0.5, dtype=x.dtype)
    const_1 = _tvm.tir.const(1, dtype=x.dtype)
    const_sqrt_2 = _tvm.tir.const(math.sqrt(2), dtype=x.dtype)
    const_sqrt_pi = _tvm.tir.const(math.sqrt(math.pi), dtype=x.dtype)
    # cdf = 0.5 * (1 + erf(x/sqrt(2)))
    cdf = const_point_5 * (const_1 + _tvm.topi.erf(x / const_sqrt_2))
    # beta = 1 / sqrt(2*pi)
    const_beta = const_1 / (const_sqrt_2 * const_sqrt_pi)
    # pdf = beta * e^(-0.5x^2)
    pdf = const_beta * _tvm.topi.exp(x * x * const_minus_point_5)
    return [dy * (cdf + x * pdf)]


_reg.register_injective_schedule("raf.op.tvm.gelu_dx")
