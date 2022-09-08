# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, unused-argument
"""Helper functions of RAF sharding system"""
from typing import List
from raf.distributed.sharding.shardspec import ShardSpec, UnsetShardSpec
from raf import distributed as dist

world_size_ = dist.get_communicator().size


def make_shard_spec(
    phy_shape: List[int], subgroup_shape: List[int] = None, ranks=world_size_, mutable=True
):
    """Create a sharding specification of a tensor.

    Parameters
    ----------
    phy_shape : List[int]
        The shape of the physical device mesh. For example, for a 2D tensor, if there are 4 devices
        in total and phy_shape is set to [2, 2] (subgrouping is not enabled), the tensor will be
        partitioned into [[x0, x1], [x2, x3]], where the device 0-3 will hold x0-3 respectively.

    subgroup_shape : Optional[List[int]]
        The shape of the subgroup. For example, for a 2D tensor, if there are 4 devices
        in total, phy_shape is set to [2, 2], subgroup_shape is set to [1, 2], the subgroup g0, g1,
        the subgroup grid will be [[d0, d1]], [[d2, d3]], [[g0], [g1]] respectively,
        and the tensor will be partitioned into [[x0], [x1]]. Since the data shard will be
        replicated within the subgroup, d0, d1 will hold x0 and d2, d3 will hold x1. To help you
        better understand this feature, it is recommended to use raf._ffi.sharding.PrintAllocTable
        to print out the detailed information of the data layout.

    ranks : Optional[Union[int, List[int]]]
        The list of ranks that participate in the computation. When ranks is set to an integer N,
        it is equivalent to [0...N-1]. By default, it will utilize all available devices specified
        by the launcher (e.g. mpirun).


    mutable: Optional[bool]
        When this flag is False, it disallows Sharding Propagation Pass to reshard this tensor.
        During the propagation, it is likely to reshard an intermediate variable to get more
        opportunities of finding a new sharding solution. It is recommended to make the specs
        of inputs and outputs immutable to get an sharding solution with expected input and
        output shape.
        Default: True.

    Returns
    -------
    spec : ShardSpec
        The created sharding specification.
    """
    if isinstance(ranks, int):
        ranks = list(range(ranks))
    if subgroup_shape is None:
        subgroup_shape = [1] * len(phy_shape)
    return ShardSpec(ranks, phy_shape, subgroup_shape, mutable)


def make_replicated_spec(ndim: int, ranks=world_size_, mutable=True):
    """Create a replicated specification of a tensor. Every device will hold a full copy of
    this tensor. Note that this is a special case of sharding specification.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the target tensor.

    ranks : Optional[Union[int, List[int]]]
        The list of ranks that participate in the computation. When ranks is set to an integer N,
        it is equivalent to [0...N-1]. By default, it will utilize all available devices specified
        by the launcher (e.g. mpirun).


    mutable: Optional[bool]
        When this flag is False, it disallows Sharding Propagation Pass to reshard this tensor.
        During the propagation, it is likely to reshard an intermediate variable to get more
        opportunities of finding a new sharding solution. It is recommended to make the specs
        of inputs and outputs immutable to get an sharding solution with expected input and
        output shape.
        Default: True.

    Returns
    -------
    spec : ShardSpec
        The created replicated specification.
    """
    if isinstance(ranks, int):
        ranks = list(range(ranks))
    phy_shape = [len(ranks)] + [1] * (ndim - 1)
    subgroup_shape = phy_shape
    return ShardSpec(ranks, phy_shape, subgroup_shape, mutable)


def make_unset_spec():
    """Create a placeholder of sharding specification of a tensor. This can be used to annotate
    the intermediate variables and denote Sharding Propagation Pass to fill in the placeholders.

    Returns
    -------
    spec : UnsetShardSpec
        The created unset sharding specification.
    """
    return UnsetShardSpec()
