# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring
"""SGD compute definition and schedule."""
from .._lib import register_compute
from .._lib import tvm as _tvm
from .._lib import _reg

_topi = _tvm.topi  # pylint: disable=invalid-name,no-member


@register_compute("raf.op.tvm.sgd")
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


_reg.register_injective_schedule("raf.op.tvm.sgd")
