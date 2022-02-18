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
"""SGD compute definition and schedule."""
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member


@register_compute("mnm.op.tvm.sgd")
def sgd_compute(attr, inputs, output_type):
    # pylint: disable=unused-argument, invalid-name
    learning_rate, mu = attr.learning_rate, attr.mu
    x0, dx, v0 = inputs
    learning_rate = _tvm.tir.const(learning_rate, dtype=x0.dtype)
    mu = _tvm.tir.const(mu, dtype=x0.dtype)

    def fcomputev(*args):
        return mu * v0(*args) + dx(*args)

    v1 = _tvm.te.compute(v0.shape, fcomputev)

    def fcomputex(*args):
        return x0(*args) - learning_rate * v1(*args)

    x1 = _tvm.te.compute(x0.shape, fcomputex)
    return [v1, x1]


_reg.register_injective_schedule("mnm.op.tvm.sgd")
