# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
