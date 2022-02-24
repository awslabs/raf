# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-function-docstring
from raf._core.ndarray import ndarray
from raf._core.value import BoolValue, FloatValue, IntValue, StringValue
from raf._lib import Array, relay


def Any(x):  # pylint: disable=invalid-name
    if isinstance(x, (IntValue, FloatValue, StringValue)):
        return x.data
    if isinstance(x, BoolValue):
        return bool(x.data)
    if isinstance(x, relay.Var):
        return ndarray(x)
    if isinstance(x, tuple):
        return tuple(map(Any, x))
    if isinstance(x, (list, Array)):
        return list(map(Any, x))
    raise NotImplementedError(type(x))


def ArrayLike(x):  # pylint: disable=invalid-name
    if isinstance(x, (IntValue, FloatValue, StringValue)):
        return x.data
    if isinstance(x, BoolValue):
        return bool(x.data)
    if isinstance(x, relay.Var):
        return ndarray(x)
    raise NotImplementedError(type(x))


def TupleTensor(x):  # pylint: disable=invalid-name
    assert isinstance(x, Array)
    return [Tensor(y) for y in x]


def Tensor(x):  # pylint: disable=invalid-name
    if isinstance(x, relay.Var):
        return ndarray(x)
    raise NotImplementedError(type(x))
