# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Matrix-multiplication operator definitions and argument data structures."""
from .base import Op, Tensor


class MatmulArgs:
    @staticmethod
    def f(x1: Tensor, x2: Tensor) -> Tensor:
        pass

    __ops__ = [
        Op("matmul"),
        Op("matmul_nt"),
        Op("matmul_tn"),
        Op("matmul_tt"),
    ]


class BatchMatmulArgs:
    @staticmethod
    def f(x1: Tensor, x2: Tensor) -> Tensor:
        pass

    __ops__ = [
        Op("batch_matmul"),
        Op("batch_matmul_nt"),
        Op("batch_matmul_tn"),
        Op("batch_matmul_tt"),
    ]
