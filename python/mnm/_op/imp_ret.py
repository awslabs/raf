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
from mnm._core.ndarray import ndarray
from mnm._core.value import BoolValue, FloatValue, IntValue, StringValue
from mnm._lib import Array, relay


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
