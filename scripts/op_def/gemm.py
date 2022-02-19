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
