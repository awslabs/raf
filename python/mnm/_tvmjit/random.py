# pylint: disable=missing-function-docstring
"""Compute definition and schedules for random operators"""
from .._lib import _reg
from .._lib import strategy

_reg.register_strategy("mnm.op.threefry_generate", strategy.threefry_generate_strategy)
_reg.register_strategy("mnm.op.threefry_split", strategy.threefry_split_strategy)
