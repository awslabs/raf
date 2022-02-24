# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import raf

TYPES = (int, float, bool)

TYPE_TABLE = {
    int: int,
    float: float,
    bool: int,
    (int, int): int,
    (int, float): float,
    (int, bool): int,
    (float, int): float,
    (float, float): float,
    (float, bool): float,
    (bool, int): int,
    (bool, float): float,
    (bool, bool): int,
}


def test_add():
    for x1, x2 in product(range(-5, 5), range(-5, 5)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.add(x1, x2)
            expected = x1 + x2
            assert result == expected
            assert isinstance(result, TYPE_TABLE[(t_1, t_2)])


def test_subtract():
    for x1, x2 in product(range(-5, 5), range(-5, 5)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.subtract(x1, x2)
            expected = x1 - x2
            assert result == expected
            assert isinstance(result, TYPE_TABLE[(t_1, t_2)])


def test_multiply():
    for x1, x2 in product(range(-5, 5), range(-5, 5)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.multiply(x1, x2)
            expected = x1 * x2
            assert result == expected
            assert isinstance(result, TYPE_TABLE[(t_1, t_2)])


def test_divide():
    for x1 in range(10, 50, 5):
        for x2 in range(1, 10):
            assert raf.divide(x1, x2) == x1 // x2
            assert raf.divide(float(x1), float(x2)) == x1 / x2


def test_mod():
    for x1 in range(10, 50, 5):
        for x2 in range(1, 10):
            assert raf.mod(x1, x2) == x1 % x2


def test_negative():
    for x1 in range(10):
        for t_1 in TYPES:
            x1 = t_1(x1)
            result = raf.negative(x1)
            expected = -x1
            assert result == expected
            assert isinstance(result, TYPE_TABLE[t_1])


def test_less():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.less(x1, x2)
            expected = x1 < x2
            assert result == expected
            assert isinstance(result, bool)


def test_greater():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.greater(x1, x2)
            expected = x1 > x2
            assert result == expected
            assert isinstance(result, bool)


def test_less_equal():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.less_equal(x1, x2)
            expected = x1 <= x2
            assert result == expected
            assert isinstance(result, bool)


def test_greater_equal():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.greater_equal(x1, x2)
            expected = x1 >= x2
            assert result == expected
            assert isinstance(result, bool)


def test_equal():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.equal(x1, x2)
            expected = x1 == x2
            assert result == expected
            assert isinstance(result, bool)


def test_not_equal():
    for x1, x2 in product(range(3), range(3)):
        for t_1, t_2 in product(TYPES, TYPES):
            x1, x2 = t_1(x1), t_2(x2)
            result = raf.not_equal(x1, x2)
            expected = x1 != x2
            assert result == expected
            assert isinstance(result, bool)


if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
    test_mod()
    test_negative()
    test_less()
    test_greater()
    test_less_equal()
    test_greater_equal()
    test_equal()
    test_not_equal()
