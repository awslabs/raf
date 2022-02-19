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
