#pylint: disable=invalid-name, protected-access, no-self-use
"""Scope with certain backends enabled"""
from __future__ import absolute_import

import threading
import tvm
from .. import _ffi

class Backend:
    """Backend scope to specify a list of preferred backends

    Parameters
    ----------
    backends : List[str]
        The list of preferred backends, with descending priority
    """
    valid_backends = ["tvm", "cuda", "cudnn", "cutlass", "cublas", "nccl"]
    storage = threading.local()
    storage.scope = []

    def __init__(self, backends):
        if not set(Backend.valid_backends).issuperset(set(backends)):
            raise ValueError(
                f"{set(backends).difference(set(Backend.valid_backends))} are not valid backends")
        self.backends = backends

    def __enter__(self):
        Backend.storage.scope.append(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        Backend.storage.scope = Backend.storage.scope[:-1]


@tvm._ffi.register_func("mnm.backend.preferred_backends")
def preferred_backends():
    """Get a list of preferred backends, with descending priority"""
    if not Backend.storage.scope:
        return None
    return Backend.storage.scope[-1].backends


class CUDNNConfig:  # pylint: disable=too-few-public-methods
    """CUDNN configuration."""

    @property
    def benchmark(self):
        """Get the benchmark flag. It controls whether to benchmark the performance when choosing
        CUDNN algorithms."""
        return _ffi.backend.cudnn.ConfigGetBenchmark()

    @benchmark.setter
    def benchmark(self, benchmark):
        """Set the benchmark flag."""
        _ffi.backend.cudnn.ConfigSetBenchmark(benchmark)


cudnn = CUDNNConfig()
