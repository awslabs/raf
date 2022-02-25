# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Other useful argument data structures."""
from .base import Any, Op, Tensor


class TensorAttrsArgs:
    @staticmethod
    def f(x: Tensor) -> Any:
        ...

    __ops__ = [
        Op("tensor.shape", namespace="__hidden__"),
        Op("tensor.dtype", namespace="__hidden__"),
        Op("tensor.device", namespace="__hidden__"),
        Op("tensor.ndim", namespace="__hidden__"),
    ]


class BroadcastRelationsArgs:
    @staticmethod
    def f(
        source: Tensor,
        target: Tensor,
    ) -> Any:
        ...

    __ops__ = [
        Op("bcast_rel.bwd_axis", namespace="__hidden__"),
        Op("bcast_rel.bwd_keepdims", namespace="__hidden__"),
    ]
