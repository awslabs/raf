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

# pylint: disable=missing-module-docstring,missing-function-docstring
import numpy as np

from mnm._core import value
from mnm._core.ndarray import Symbol
from mnm._core.ndarray import get_ndarray_handle as nd_handle
from mnm._core.ndarray import get_symbol_handle as handle
from mnm._core.ndarray import ndarray


def Int(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, int):
        raise TypeError("Cannot convert to int")
    return a


def Double(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, (int, float)):
        raise TypeError("Cannot convert to float")
    return float(a)


def String(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, str):
        raise TypeError("Cannot convert to str")
    return a


def Bool(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, (int, bool)):
        raise TypeError("Cannot convert to bool")
    return bool(a)


def Device(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, str):
        raise TypeError("Cannot convert to device")
    return a


def DType(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, str):
        raise TypeError("Cannot convert to dtype")
    return a


def Tensor(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if isinstance(a, ndarray):
        return nd_handle(a)
    if a is None:
        return None
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return value.Value.as_const_expr(value.TensorValue.from_numpy(a))


def ArrayLike(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if isinstance(a, ndarray):
        return nd_handle(a)
    if a is None:
        return None
    if isinstance(a, (int, float, str)):
        return a
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return value.Value.as_const_expr(value.TensorValue.from_numpy(a))


def TupleInt(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
    if not isinstance(a, (list, tuple)):
        raise TypeError("Cannot convert to Tuple[int]")
    if isinstance(a, list):
        a = tuple(a)
    for item in a:
        if not isinstance(item, int):
            raise TypeError("Cannot convert to Tuple[int]")
    return a


def IntOrTupleInt(a):  # pylint: disable=invalid-name
    if isinstance(a, Symbol):
        return handle(a)
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
    if isinstance(a, Symbol):
        return handle(a)
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
    if isinstance(a, Symbol):
        return handle(a)
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
