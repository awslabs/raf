# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-function-docstring
import numpy as np

from raf._core.ndarray import get_ndarray_handle as handle
from raf._core.ndarray import ndarray


def Int(a):  # pylint: disable=invalid-name
    if not isinstance(a, int):
        raise TypeError("Cannot convert to int")
    return a


def Double(a):  # pylint: disable=invalid-name
    if not isinstance(a, (int, float)):
        raise TypeError("Cannot convert to float")
    return float(a)


def String(a):  # pylint: disable=invalid-name
    if not isinstance(a, str):
        raise TypeError("Cannot convert to str")
    return a


def Bool(a):  # pylint: disable=invalid-name
    if not isinstance(a, (int, bool)):
        raise TypeError("Cannot convert to bool")
    return bool(a)


def Device(a):  # pylint: disable=invalid-name
    if not isinstance(a, str):
        raise TypeError("Cannot convert to device")
    return a


def DType(a):  # pylint: disable=invalid-name
    if not isinstance(a, str):
        raise TypeError("Cannot convert to dtype")
    return a


def Tensor(a):  # pylint: disable=invalid-name
    if a is None:
        return None
    if isinstance(a, ndarray):
        return handle(a)
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return handle(ndarray(a))


def ArrayLike(a):  # pylint: disable=invalid-name
    if a is None:
        return None
    if isinstance(a, (int, float, str)):
        return a
    if isinstance(a, ndarray):
        return handle(a)
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return handle(ndarray(a))


def TupleInt(a):  # pylint: disable=invalid-name
    if not isinstance(a, (list, tuple)):
        raise TypeError("Cannot convert to Tuple[int]")
    if isinstance(a, list):
        a = tuple(a)
    for item in a:
        if not isinstance(item, int):
            raise TypeError("Cannot convert to Tuple[int]")
    return a


def IntOrTupleInt(a):  # pylint: disable=invalid-name
    if isinstance(a, int):
        return a
    if not isinstance(a, (list, tuple)):
        raise TypeError("Cannot convert to Union[int, Tuple[int]]")
    if isinstance(a, list):
        a = tuple(a)
    for item in a:
        if not isinstance(item, int):
            raise TypeError("Cannot convert to Union[int, Tuple[int]")
    return a


def IntOrTupleIntOrNone(a):  # pylint: disable=invalid-name
    if a is None or isinstance(a, int):
        return a
    if not isinstance(a, (list, tuple)):
        raise TypeError("Cannot convert to Optional[Union[int, Tuple[int]]]")
    if isinstance(a, list):
        a = tuple(a)
    for item in a:
        if not isinstance(item, int):
            raise TypeError("Cannot convert to Optional[Union[int, Tuple[int]]]")
    return a


def BoolOrTupleInt(a):  # pylint: disable=invalid-name
    if isinstance(a, bool):
        return a
    if not isinstance(a, (list, tuple)):
        raise TypeError("Cannot convert to Union[bool, Tuple[int]]")
    if isinstance(a, list):
        a = tuple(a)
    for item in a:
        if not isinstance(item, int):
            raise TypeError("Cannot convert to Union[bool, Tuple[int]]")
    return a
