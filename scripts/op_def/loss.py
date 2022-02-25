# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Loss function definition and their argument data structures."""
from .base import Op, Tensor


class LossArgs:

    __ops__ = [
        Op("smooth_l1_loss"),
        Op("smooth_l1_loss_dtrue"),
        Op("smooth_l1_loss_dpred"),
        Op("nll_loss"),
        Op("nll_loss_dtrue"),
        Op("nll_loss_dpred"),
        Op("cross_entropy"),
        Op("cross_entropy_dtrue"),
        Op("cross_entropy_dpred"),
    ]

    @staticmethod
    def f(
        y_true: Tensor,
        y_pred: Tensor,
    ) -> Tensor:
        ...
