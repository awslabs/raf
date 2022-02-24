# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from raf import hybrid


@hybrid
def tuple1():
    a = (1, 2, 3)
    b, c, d = a
    return b + c + d


@hybrid
def tuple2():
    a = ((1, 2), 3)
    (b, c), d = a
    return b + c + d


@hybrid
def tuple3():
    a = ((1, 2), 3)
    b, c = a
    d, e = b
    return d + e + c


def tuple4():
    a = (1, 2, 3)
    b = a[0:2]
    c = b[0:1]
    d = c[0]
    return d


def test_tuple_1():
    assert tuple1() == 6


def test_tuple_2():
    assert tuple2() == 6


def test_tuple_3():
    assert tuple3() == 6


def test_tuple_4():
    assert tuple4() == 1


if __name__ == "__main__":
    test_tuple_1()
    test_tuple_2()
    test_tuple_3()
    test_tuple_4()
