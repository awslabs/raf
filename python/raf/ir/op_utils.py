# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-function-docstring
"""Graph IR builder utilities for ops"""
from numbers import Number

from raf._lib import relay
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
