# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute definition and schedules for data init operators"""
from .._lib import _reg

_reg.register_injective_schedule("raf.op.tvm.zeros")
_reg.register_injective_schedule("raf.op.tvm.ones")
_reg.register_injective_schedule("raf.op.tvm.one_hot")
