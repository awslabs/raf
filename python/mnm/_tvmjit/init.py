# pylint: disable=missing-function-docstring
"""Compute definition and schedules for data init operators"""
from .._lib import _reg

_reg.register_injective_schedule("mnm.op.zeros")
_reg.register_injective_schedule("mnm.op.ones")
_reg.register_injective_schedule("mnm.op.one_hot")
