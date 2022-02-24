# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring
"""Compute definition and schedules for vision functions."""
from .nn import schedule_generic
from .._lib import _reg
from .._lib import strategy, register_compute
from .._lib import tvm as _tvm

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member

_reg.register_strategy("raf.op.tvm.get_valid_counts", strategy.get_valid_counts_strategy)
_reg.register_strategy("raf.op.tvm.non_max_suppression", strategy.nms_strategy)
_reg.register_strategy("raf.op.tvm.roi_align", strategy.roi_align_strategy)


@register_compute("raf.op.tvm.roi_align_dx")
def roi_align_dx_compute(attrs, inputs, output_type):
    # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    data = inputs[0]
    rois = inputs[1]
    dy = inputs[2]
    pooled_size, spatial_scale, sample_ratio, layout, mode = (
        attrs.pooled_size,
        attrs.spatial_scale,
        attrs.sample_ratio,
        attrs.layout,
        attrs.mode,
    )
    pooled_size = _topi.utils.get_const_tuple(pooled_size)
    mode = bytes(mode, encoding="utf-8")
    if layout == "NCHW":
        R = _topi.vision.roi_align_nchw(data, rois, pooled_size, spatial_scale, mode, sample_ratio)
    else:
        R = _topi.vision.roi_align_nhwc(data, rois, pooled_size, spatial_scale, mode, sample_ratio)
    grads = _tvm.te.gradient(R, [data], head=dy)
    return grads


_reg.register_schedule("raf.op.tvm.roi_align_dx", schedule_generic)
