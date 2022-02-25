# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule registries for broadcast operators."""
from .._lib import _reg

_reg.register_broadcast_schedule("raf.op.tvm.add")
_reg.register_broadcast_schedule("raf.op.tvm.subtract")
_reg.register_broadcast_schedule("raf.op.tvm.multiply")
_reg.register_broadcast_schedule("raf.op.tvm.divide")
_reg.register_broadcast_schedule("raf.op.tvm.floor_divide")
_reg.register_broadcast_schedule("raf.op.tvm.maximum")
_reg.register_broadcast_schedule("raf.op.tvm.minimum")
_reg.register_broadcast_schedule("raf.op.tvm.bias_add")
_reg.register_broadcast_schedule("raf.op.tvm.power")
_reg.register_broadcast_schedule("raf.op.tvm.where")
_reg.register_broadcast_schedule("raf.op.tvm.logical_and")
_reg.register_broadcast_schedule("raf.op.tvm.right_shift")
_reg.register_broadcast_schedule("raf.op.tvm.left_shift")
_reg.register_broadcast_schedule("raf.op.tvm.equal")
_reg.register_broadcast_schedule("raf.op.tvm.not_equal")
_reg.register_broadcast_schedule("raf.op.tvm.less")
_reg.register_broadcast_schedule("raf.op.tvm.less_equal")
_reg.register_broadcast_schedule("raf.op.tvm.greater")
_reg.register_broadcast_schedule("raf.op.tvm.greater_equal")
