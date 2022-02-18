# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=missing-function-docstring
"""Compute definition and schedules for vision functions."""
from .nn import schedule_generic
from .._lib import _reg
from .._lib import strategy, register_compute
from .._lib import tvm as _tvm

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member

_reg.register_strategy("mnm.op.tvm.get_valid_counts", strategy.get_valid_counts_strategy)
_reg.register_strategy("mnm.op.tvm.non_max_suppression", strategy.nms_strategy)
_reg.register_strategy("mnm.op.tvm.roi_align", strategy.roi_align_strategy)


@register_compute("mnm.op.tvm.roi_align_dx")
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


_reg.register_schedule("mnm.op.tvm.roi_align_dx", schedule_generic)
