# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=missing-function-docstring, missing-module-docstring
# pylint: disable=unused-argument, invalid-name

"""Schedule registries for broadcast operators."""
from .._lib import tvm as _tvm
from .._lib import _reg, register_compute

_topi = _tvm.topi  # pylint: disable=no-member

_reg.register_broadcast_schedule("raf.op.tvm.add")
_reg.register_broadcast_schedule("raf.op.tvm.subtract")
_reg.register_broadcast_schedule("raf.op.tvm.multiply")
_reg.register_broadcast_schedule("raf.op.tvm.divide")
_reg.register_broadcast_schedule("raf.op.tvm.floor_divide")
_reg.register_broadcast_schedule("raf.op.tvm.maximum")
_reg.register_broadcast_schedule("raf.op.tvm.minimum")
_reg.register_broadcast_schedule("raf.op.tvm.bias_add")
_reg.register_broadcast_schedule("raf.op.tvm.power")
_reg.register_broadcast_schedule("raf.op.tvm.logical_and")
_reg.register_broadcast_schedule("raf.op.tvm.right_shift")
_reg.register_broadcast_schedule("raf.op.tvm.left_shift")
_reg.register_broadcast_schedule("raf.op.tvm.equal")
_reg.register_broadcast_schedule("raf.op.tvm.not_equal")
_reg.register_broadcast_schedule("raf.op.tvm.less")
_reg.register_broadcast_schedule("raf.op.tvm.less_equal")
_reg.register_broadcast_schedule("raf.op.tvm.greater")
_reg.register_broadcast_schedule("raf.op.tvm.greater_equal")


@register_compute("raf.op.tvm.where")
def compute_where(attr, inputs, output_type):
    cond, x1, x2 = inputs

    if x1.dtype != x2.dtype:
        raise ValueError("x1 and x2 has different dtypes: %s vs %s" % (x1.dtype, x2.dtype))
    dtype = x1.dtype

    if cond.dtype != dtype:
        cond = _topi.cast(cond, dtype)

    out = x1 * cond + x2 * (1 - cond)
    return [out]


_reg.register_broadcast_schedule("raf.op.tvm.where")
