# pylint: disable=missing-function-docstring
"""Compute definition and schedules for vision functions."""
from .._lib import _reg
from .._lib import strategy

_reg.register_strategy("mnm.op.get_valid_counts", strategy.get_valid_counts_strategy)
_reg.register_strategy("mnm.op.non_max_suppression", strategy.nms_strategy)
