# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-function-docstring
from numbers import Number

import numpy as np

from raf._core.ndarray import ndarray
from raf._core.value import BoolValue, FloatValue, IntValue, StringValue, TensorValue, Value
from raf._lib import Array, relay


def to_any(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if a is None:
        return None
    if isinstance(a, (list, tuple)):
        return to_int_tuple(a)
    if isinstance(a, (Number, str)):
        return a
    return to_tensor(a)


def to_tensor(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if a is None:
        return None
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    # TODO(@junrushao1994): save this FFI call
    return Value.as_const_expr(TensorValue.from_numpy(a))


def to_int_tuple(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if a is None:
        a = []
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(a, Number):
        if int(a) != a:
            raise ValueError("Cannot convert to List[int]")
        return int(a)
    if not isinstance(a, (tuple, list)):
        raise ValueError("Cannot convert to List[int]")
    result = []
    for item in a:
        if isinstance(item, Number) and int(item) == item:
            result.append(int(item))
        else:
            raise ValueError("Cannot convert to List[int]")
    return result


def to_optional_int_tuple(a):
    return None if a is None else to_int_tuple(a)


def to_tensor_tuple(a):
    if not isinstance(a, (tuple, list)):
        raise ValueError(f"Cannot convert {a} to List[tensor]")
    result = []
    for item in a:
        if isinstance(item, ndarray):
            result.append(item._ndarray__handle)  # pylint: disable=protected-access
        else:
            raise ValueError(f"Cannot convert {a} to List[tensor]")
    return result


def to_int(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()
    if isinstance(a, Number) and int(a) == a:
        return int(a)
    raise ValueError("Cannot convert to int")


def to_double(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()
    if isinstance(a, Number) and float(a) == a:
        return float(a)
    raise ValueError("Cannot convert to double")


def to_bool(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim <= 1:
        a = a.item()
    if isinstance(a, Number) and bool(a) == a:
        return bool(a)
    raise ValueError("Cannot convert to bool")


def to_string(a):
    if isinstance(a, ndarray):
        return a._ndarray__handle  # pylint: disable=protected-access
    if isinstance(a, str):
        return a
    raise ValueError("Cannot convert to str")


def ret(a):
    if isinstance(a, (IntValue, FloatValue, StringValue)):
        return a.value
    if isinstance(a, BoolValue):
        return bool(a.value)
    if isinstance(a, relay.Var):
        return ndarray(a)
    if isinstance(a, tuple):
        return tuple(map(ret, a))
    if isinstance(a, list):
        return list(map(ret, a))
    if isinstance(a, Array):
        return list(map(ret, a))
    raise NotImplementedError(type(a))
