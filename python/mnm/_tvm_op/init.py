"""Compute definition and schedules for data init operators"""
from .._lib import _reg

_reg.register_injective_schedule("mnm.op.tvm.zeros")
_reg.register_injective_schedule("mnm.op.tvm.ones")
_reg.register_injective_schedule("mnm.op.tvm.one_hot")
