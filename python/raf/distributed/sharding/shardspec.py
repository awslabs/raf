# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, unused-argument
"""Definition of RAF sharding specifications and attributes."""
from raf._core.core_utils import register_node
from raf._ffi.sharding import _make
from raf._core.value import Value
from tvm.ir.container import Array


@register_node("raf.sharding.BaseShardSpec")
class BaseShardSpec(Value):
    """Base type of Sharding Specifications"""


@register_node("raf.sharding.ShardSpec")
class ShardSpec(BaseShardSpec):
    """Sharding Specifications"""

    mutable: int
    ndim: int
    nshard: int
    ngroup: int
    ranks: Array
    logic_shape: Array
    logic_index: Array
    phy_shape: Array
    phy_index: Array
    subgroup_shape: Array
    subgroup_index: Array

    def __init__(self, ranks, phy_shape, subgroup_shape, mutable):
        self.__init_handle_by_constructor__(
            _make.ShardSpec, ranks, phy_shape, subgroup_shape, mutable
        )

    def make_updated(self, ranks=None, phy_shape=None, subgroup_shape=None, mutable=None):
        """Make a new spec based on this spec with a few fields modified"""
        ranks = ranks if ranks else self.ranks
        phy_shape = phy_shape if phy_shape else self.phy_shape
        subgroup_shape = subgroup_shape if subgroup_shape else self.subgroup_shape
        mutable = mutable if mutable else self.mutable
        return ShardSpec(ranks, phy_shape, subgroup_shape, mutable)


@register_node("raf.sharding.UnsetShardSpec")
class UnsetShardSpec(BaseShardSpec):
    """Placeholder of Sharding Specifications"""

    def __init__(self):
        self.__init_handle_by_constructor__(_make.UnsetShardSpec)
