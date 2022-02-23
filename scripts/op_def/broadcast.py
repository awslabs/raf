# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods,too-many-arguments,unused-argument,invalid-name
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Broadcast operator definition and their argument data structures."""
from .base import Op, ArrayLike


class UnaryArgs:
    @staticmethod
    def f(x: ArrayLike) -> ArrayLike:
        ...

    __ops__ = [
        Op("copy"),
        Op("tanh"),
        Op("abs"),
        Op("ceil"),
        Op("cos"),
        Op("floor"),
        Op("log"),
        Op("exp"),
        Op("sigmoid"),
        Op("negative"),
        Op("logical_not"),
        Op("relu"),
    ]


class UnaryDxArgs:
    @staticmethod
    def f(
        y: ArrayLike,
        dy: ArrayLike,
        x: ArrayLike,
    ) -> ArrayLike:
        ...

    __ops__ = [
        Op("relu_dx"),
        Op("tanh_dx"),
        Op("sigmoid_dx"),
    ]


class BinaryArgs:
    @staticmethod
    def f(
        x1: ArrayLike,
        x2: ArrayLike,
    ) -> ArrayLike:
        ...

    __ops__ = [
        Op("add"),
        Op("subtract"),
        Op("multiply"),
        Op("divide"),
        Op("mod"),
        Op("less"),
        Op("greater"),
        Op("less_equal"),
        Op("greater_equal"),
        Op("equal"),
        Op("not_equal"),
    ]


# TODO(@junrushao1994): implement `ufunc`s
