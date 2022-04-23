# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
"""Distributed Context"""
import raf._ffi.distributed as ffi
from raf._core.core_utils import register_node
from raf._lib import Object
from raf.build import with_mpi


@register_node("raf.distributed.Communicator")
class Communicator(Object):
    pass


if with_mpi():

    @register_node("raf.distributed.MPICommunicator")
    class MPICommunicator(Object):
        pass


@register_node("raf.distributed.VoidCommunicator")
class VoidCommunicator(Communicator):
    @property
    def size(self):
        return self.size_

    @size.setter
    def size(self, value):
        self.size_ = value
        ffi.SetGlobalSize(value)

    @property
    def rank(self):
        return self.rank_

    @rank.setter
    def rank(self, value):
        self.rank_ = value
        ffi.SetGlobalRank(value)

    @property
    def local_size(self):
        return self.local_size_

    @local_size.setter
    def local_size(self, value):
        self.local_size_ = value
        ffi.SetGlobalLocalSize(value)

    @property
    def local_rank(self):
        return self.local_rank_

    @local_rank.setter
    def local_rank(self, value):
        self.local_rank_ = value
        ffi.SetGlobalLocalRank(value)

    def dumps(self):
        attr_keys = [
            "size",
            "rank",
            "local_size",
            "local_rank",
        ]
        return {attr: getattr(self, attr) for attr in attr_keys}

    def loads(self, context_dict):
        for attr in context_dict:
            setattr(self, attr, context_dict[attr])


def get_communicator():
    return ffi.GetGlobalCommunicator()


def set_default_communicator(name):
    assert name in ["mpi", "void"], "Invalid name to set global communicator!"
    ffi.SetDefaultCommunicator(name)
