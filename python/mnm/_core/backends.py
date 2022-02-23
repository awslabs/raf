# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, protected-access, no-self-use
"""Scope with certain backends enabled"""
from __future__ import absolute_import

from .. import _ffi


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
