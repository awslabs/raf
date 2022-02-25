# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""NN-specific operators and their argument data structures."""
from .base import IntOrTupleInt, IntOrTupleIntOrNone, Op, Tensor, Tuple, TupleInt


class ConvArgs:

    __ops__ = [
        # Op("conv1d"),
        Op("conv2d"),
        # Op("conv3d"),
    ]

    @staticmethod
    def f(
        x: Tensor,
        w: Tensor,
        stride: IntOrTupleInt = 1,
        padding: IntOrTupleInt = 0,
        dilation: IntOrTupleInt = 1,
        group: int = 1,
    ) -> Tensor:
        ...


class ConvDxDwArgs:

    __ops__ = [
        # Op("conv1d_dx"),
        # Op("conv1d_dw"),
        Op("conv2d_dx"),
        Op("conv2d_dw"),
        # Op("conv3d_dx"),
        # Op("conv3d_dw"),
    ]

    @staticmethod
    def f(
        y: Tensor,
        dy: Tensor,
        x_or_w: Tensor,
        shape: TupleInt,
        stride: IntOrTupleInt = 1,
        padding: IntOrTupleInt = 0,
        dilation: IntOrTupleInt = 1,
        groups: int = 1,
    ) -> Tensor:
        ...


class PoolArgs:

    __ops__ = [
        # Op("max_pool1d"),
        Op("max_pool2d"),
        # Op("max_pool3d"),
        # Op("avg_pool1d"),
        Op("avg_pool2d"),
        # Op("avg_pool3d"),
    ]

    @staticmethod
    def f(
        x: Tensor,
        kernel: IntOrTupleInt,
        stride: IntOrTupleIntOrNone = None,
        padding: IntOrTupleInt = 0,
        dilation: IntOrTupleInt = 1,
        ceil_mode: bool = False,
        include_pad: bool = True,
    ) -> Tensor:
        ...


class PoolDxArgs:

    __ops__ = [
        # Op("max_pool1d_dx"),
        Op("max_pool2d_dx"),
        # Op("max_pool3d_dx"),
        # Op("avg_pool1d_dx"),
        Op("avg_pool2d_dx"),
        # Op("avg_pool3d_dx"),
    ]

    @staticmethod
    def f(
        y: Tensor,
        dy: Tensor,
        x: Tensor,
        kernel: IntOrTupleInt,
        stride: IntOrTupleIntOrNone = None,
        padding: IntOrTupleInt = 0,
        dilation: IntOrTupleInt = 1,
        ceil_mode: bool = False,
        include_pad: bool = True,
    ) -> Tensor:
        ...


class SoftmaxArgs:

    __ops__ = [
        Op("softmax"),
        Op("log_softmax"),
    ]

    @staticmethod
    def f(
        x: Tensor,
        axis: IntOrTupleInt = -1,
    ) -> Tensor:
        ...


class SoftmaxDxArgs:

    __ops__ = [
        Op("softmax_dx"),
        Op("log_softmax_dx"),
    ]

    @staticmethod
    def f(
        y: Tensor,
        dy: Tensor,
        x: Tensor,
        axis: IntOrTupleInt = -1,
    ) -> Tensor:
        ...


class BatchNormTrainArgs:

    __ops__ = [
        Op("batch_norm_train"),
    ]

    @staticmethod
    def f(
        x: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        w: Tensor,
        b: Tensor,
        momentum: float = 0.1,
        epsilon: float = 1e-5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ...


class BatchNormInferArgs:

    __ops__ = [
        Op("batch_norm_infer"),
    ]

    @staticmethod
    def f(
        x: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        w: Tensor,
        b: Tensor,
        momentum: float = 0.1,
        epsilon: float = 1e-5,
    ) -> Tensor:
        ...


class BatchNormDxwbArgs:

    __ops__ = [
        Op("batch_norm_train_dxwb"),
    ]

    @staticmethod
    def f(
        y: Tensor,
        dy: Tensor,
        x: Tensor,
        w: Tensor,
        b: Tensor,
        epsilon: float = 1e-5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ...
