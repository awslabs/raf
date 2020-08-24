# pylint: disable=missing-class-docstring,missing-function-docstring
"""Distributed Context"""
from mnm._core.core_utils import set_module
from mnm._ffi.distributed import GetRootRank, GetRank, GetSize
from mnm._ffi.distributed import GetLocalRank, GetLocalSize, RemoveCommunicator


@set_module("mnm")  # pylint: disable=invalid-name,too-many-instance-attributes
class DistContext:
    ctx = None

    def __init__(self):
        self.root_rank = GetRootRank()
        self.rank = GetRank()
        self.size = GetSize()
        self.local_rank = GetLocalRank()
        self.local_size = GetLocalSize()

    def __del__(self):
        RemoveCommunicator()

    @staticmethod
    def get_context():
        if DistContext.ctx is None:
            DistContext.ctx = DistContext()
        return DistContext.ctx
