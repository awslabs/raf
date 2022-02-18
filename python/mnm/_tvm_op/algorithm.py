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

# pylint: disable=missing-function-docstring
"""Compute definition and schedules for vision functions."""
from .._lib import topi as _topi  # pylint: disable=unused-import
from .._lib import _reg
from .._lib import strategy

_reg.register_strategy("mnm.op.tvm.argsort", strategy.argsort_strategy)
_reg.register_strategy("mnm.op.tvm.sort", strategy.sort_strategy)
_reg.register_strategy("mnm.op.tvm.topk", strategy.topk_strategy)
