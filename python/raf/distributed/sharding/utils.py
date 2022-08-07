"""Helper functions of RAF sharding system"""

from ctypes import Union
from typing import List
from raf.distributed.sharding.shardspec import BaseShardSpec, ShardSpec, UnsetShardSpec
from raf import distributed as dist

world_size_ = dist.get_communicator().size

def make_shard_spec(phy_shape: List[int], subgroup_shape: List[int] = None, ranks = world_size_, mutable = True):
    if type(ranks) is int:
        ranks = [i for i in range(ranks)]
    if subgroup_shape is None:
        subgroup_shape = [1] * len(phy_shape)
    return ShardSpec(ranks, phy_shape, subgroup_shape, mutable)

def make_replicated_spec(ndim: int, ranks = world_size_, mutable = True):
    if type(ranks) is int:
        ranks = [i for i in range(ranks)]
    phy_shape = [len(ranks)] + [1] * (ndim - 1)
    subgroup_shape = phy_shape
    return ShardSpec(ranks, phy_shape, subgroup_shape, mutable)

def make_unset_spec():
    return UnsetShardSpec()