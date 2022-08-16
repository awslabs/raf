# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAF sharding system"""
from raf._ffi.sharding._make import ShardOpCallAttrs
from .shardspec import BaseShardSpec, ShardSpec, UnsetShardSpec
from .utils import make_replicated_spec, make_shard_spec, make_unset_spec
