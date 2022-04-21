# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for distributed training, e.g., collective communication operators."""
from raf._ffi.distributed import RemoveCommunicator
from .op import (
    allreduce,
    allgather,
    reduce,
    reduce_scatter,
    broadcast,
    send,
    recv,
    group_allgather,
    group_reduce_scatter,
)
from .config import DistConfig, get_config
from .communicator import get_communicator, set_default_communicator
