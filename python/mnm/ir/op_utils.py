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
"""Graph IR builder utilities for ops"""
from numbers import Number

from mnm._lib import relay
from .constant import const


def to_any(a):
    if isinstance(a, relay.Expr):
        return a

    if a is None:
        return const(None)

    if isinstance(a, (Number, str)):
        return const(a)

    if isinstance(a, (list, tuple)):
        try:
            return const(a)
        except:  # pylint: disable=bare-except
            ret = []
            for i in a:
                if isinstance(i, relay.Expr):
                    ret.append(i)
                else:
                    ret.append(const(i))
            return relay.Tuple(ret)

    raise ValueError(f"Cannot convert {a} to relay.Expr")


def to_tensor(a):
    if isinstance(a, relay.Expr):
        return a

    if a is None:
        return const(None)

    raise ValueError(f"Cannot convert {a} to relay.Expr")


def to_int_tuple(a):
    if isinstance(a, relay.Expr):
        return a

    if a is None:
        a = []

    if isinstance(a, Number):
        if int(a) != a:
            raise ValueError(f"Cannot convert {a} to List[int]")
        a = [a]

    if not isinstance(a, (tuple, list)):
        raise ValueError(f"Cannot convert {a} to List[int]")
    result = []

    for item in a:
        if isinstance(item, Number) and int(item) == item:
            result.append(int(item))
        else:
            raise ValueError(f"Cannot convert {a} to List[int]")

    return const(result)


def to_tensor_tuple(a):
    if isinstance(a, relay.Expr):
        return a

    if not isinstance(a, (tuple, list)):
        raise ValueError(f"Cannot convert {a} to List[relay.Expr]")
    result = []

    for item in a:
        if isinstance(item, relay.Expr):
            result.append(item)
        else:
            raise ValueError(f"Cannot convert {a} to List[relay.Expr]")

    return relay.Tuple(result)


def to_optional_int_tuple(a):
    return const(None) if a is None else to_int_tuple(a)


def to_int(a):
    if isinstance(a, relay.Expr):
        return a

    if isinstance(a, Number) and int(a) == a:
        return const(int(a))
    raise ValueError(f"Cannot convert {a} to int")


def to_double(a):
    if isinstance(a, relay.Expr):
        return a

    if isinstance(a, Number) and float(a) == a:
        return const(float(a))
    raise ValueError(f"Cannot convert {a} to double")


def to_bool(a):
    if isinstance(a, relay.Expr):
        return a

    if isinstance(a, Number) and bool(a) == a:
        return const(bool(a))
    raise ValueError(f"Cannot convert {a} to bool")


def to_string(a):
    if isinstance(a, relay.Expr):
        return a

    if isinstance(a, str):
        return const(a)
    raise ValueError(f"Cannot convert {a} to str")
