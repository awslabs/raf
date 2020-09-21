# pylint: disable=missing-function-docstring
"""Compute definition and schedules for vision functions."""
from .._lib import topi as _topi  # pylint: disable=unused-import
from .._lib import _reg
from .._lib import strategy

_reg.register_strategy("mnm.op.argsort", strategy.argsort_strategy)
