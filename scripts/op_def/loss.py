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
