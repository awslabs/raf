# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-function-docstring
from raf._core.ndarray import Symbol


def Any(x):  # pylint: disable=invalid-name
    return Symbol.from_expr(x)
