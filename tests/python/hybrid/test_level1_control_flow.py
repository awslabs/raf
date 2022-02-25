# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from raf import hybrid

# pylint: disable=all


@hybrid
def program_0(x1):
    g1 = -1
    pass
    x1 = (10 + x1) + ((2 + x1) * (5 + x1))
    x1 = (x1) + ((3 * x1) * (10 + x1))
    g1 = 0
    while g1 < 1:
        pass
        x1 = ((x1) - (10 + g1)) - (g1 // 5)
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    pass
    return x1


def test_program_0():
    ans = program_0(7)
    assert ans == 50740


@hybrid
def program_1(x1, x2, g1):
    x3 = -1
    pass
    x1 = (x1 + 2) * ((g1 % 5) - (-x2))
    x1 = (3 * x2) + ((x1 % 10) - (-g1))
    if (x1 % 2) * ((g1 + 2) * (g1)) < 10:
        pass
        x1 = ((2 + x2) - (x2)) - (x1)
        x1 = ((g1) - (10 + g1)) * (2 * x1)
        return x1
    else:
        pass
        x1 = (2 + x2) * ((x1) * (10 + g1))
        while g1 < 10:
            pass
            x1 = ((5 + x1) * (-x1)) * (3 * g1)
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        x2 = ((2 * x1) + (2 + g1)) - (-g1)
    x3 = ((-g1) + (3 * g1)) * (2 * x1)
    x3 = ((5 + x2) - (2 * x2)) - (g1 % 5)
    return x3


def test_program_1():
    ans = program_1(9, 10, 0)
    assert ans == 560


@hybrid
def program_2(x1, x2):
    x3 = -1
    g1 = -1
    g2 = -1
    g3 = -1
    g4 = -1
    pass
    x3 = (10 + x1) + ((-x1) - (3 * x2))
    x3 = (2 + x1) * ((-x3) + (2 + x1))
    g1 = 0
    while g1 < 4:
        pass
        x2 = (x2 % 5) * ((2 * x2) - (-x3))
        x2 = ((x3 % 2) * (g1 + 2)) - (x1 + 10)
        g2 = 0
        while g2 < 7:
            pass
            pass
            if (g1 // 2) + ((x1) - (x1 + 10)) < 10:
                pass
                x2 = (x2 + 10) - ((2 + x2) * (g1))
                break
            else:
                pass
                x1 = (g2 * 2) * ((5 + x2) - (x2))
                x3 = ((x3 + 5) - (5 + x3)) + (-x1)
                g3 = 0
                while g3 < 7:
                    pass
                    x2 = ((g2) - (-g2)) - (2 + x3)
                    g4 = 0
                    while g4 < 7:
                        pass
                        pass
                        break
                        g4 = g4 + 1
                    pass
                    g3 = g3 + 1
                x3 = ((x3 + 10) * (2 * x3)) * (2 + x2)
            pass
            g2 = g2 + 1
        x2 = (g2) - ((2 * x2) - (5 + x3))
        g1 = g1 + 1
    pass
    return x3


def test_program_2():
    ans = program_2(10, 8)
    assert ans == 312


@hybrid
def program_3(x1, g1):
    x2 = -1
    x3 = -1
    pass
    x2 = (-x1) + ((3 * g1) * (3 * x1))
    x1 = (x1) * ((-x1) * (10 + x1))
    while g1 < 1:
        pass
        pass
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    x1 = (2 + g1) + ((2 + x2) - (2 * x2))
    x3 = ((x1 * 3) * (2 + x1)) + (x2)
    return x3


def test_program_3():
    ans = program_3(6, 0)
    assert ans == 423


@hybrid
def program_4(x1, x2):
    x3 = -1
    pass
    x2 = ((5 + x2) * (x1 % 5)) * (10 + x1)
    x3 = (x2 // 2) + ((10 + x2) - (5 + x2))
    if ((2 * x1) + (-x3)) + (10 + x1) < 10:
        pass
        pass
        return x2
    x1 = ((2 * x1) * (x1 + 2)) + (2 * x3)
    return x2


def test_program_4():
    ans = program_4(8, 6)
    assert ans == 594


@hybrid
def program_5(x1):
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x2 = ((5 + x1) + (x1 + 5)) * (10 + x1)
    x2 = ((x1 // 2) - (2 * x2)) + (x2)
    g1 = 0
    while g1 < 10:
        pass
        x1 = (2 * x1) * ((2 + g1) + (10 + x1))
        x2 = ((g1) + (x2 % 2)) - (-g1)
        break
        g1 = g1 + 1
    x3 = (3 * x2) * ((3 * x1) * (2 + x2))
    x3 = ((x1) * (x2)) - (5 + g1)
    return x1


def test_program_5():
    ans = program_5(6)
    assert ans == 216


@hybrid
def program_6(x1):
    x2 = -1
    g1 = -1
    pass
    x1 = (x1 // 5) - ((2 * x1) * (2 + x1))
    x1 = ((x1 % 2) + (x1 // 10)) + (-x1)
    g1 = 0
    while g1 < 6:
        pass
        x2 = ((x1 % 2) - (10 + x1)) - (x1 % 5)
        break
        g1 = g1 + 1
    x1 = ((x1 % 2) + (10 + x1)) + (2 * g1)
    return x1


def test_program_6():
    ans = program_6(7)
    assert ans == 122


@hybrid
def program_7(x1):
    pass
    x1 = ((2 + x1) * (2 + x1)) - (3 * x1)
    x1 = (-x1) + ((x1 % 5) * (-x1))
    if ((10 + x1) - (2 + x1)) * (2 * x1) < 10:
        pass
        x1 = (5 + x1) * ((x1) + (10 + x1))
        return x1
    x1 = (10 + x1) * ((x1) * (5 + x1))
    return x1


def test_program_7():
    ans = program_7(9)
    assert ans == 432450


@hybrid
def program_8(x1, g1):
    g2 = -1
    pass
    x1 = ((x1) + (-g1)) - (3 * g1)
    g2 = 0
    while g2 < 6:
        pass
        x1 = (x1 % 10) - ((g2 + 2) + (3 * g1))
        break
        g2 = g2 + 1
    x1 = ((x1 * 2) * (5 + x1)) * (g2 + 5)
    x1 = ((-g2) * (5 + g1)) + (3 * x1)
    return x1


def test_program_8():
    ans = program_8(4, 0)
    assert ans == 420


@hybrid
def program_9(x1):
    g1 = -1
    pass
    x1 = (-x1) - ((-x1) * (2 + x1))
    x1 = (2 + x1) - ((x1 % 2) - (10 + x1))
    if (x1 * 2) - ((5 + x1) + (-x1)) < 10:
        pass
        pass
        if (10 + x1) - ((-x1) * (5 + x1)) < 10:
            pass
            x1 = (x1 % 5) * ((x1 + 10) - (2 * x1))
            g1 = 0
            while g1 < 4:
                pass
                x1 = (-x1) - ((g1) + (5 + g1))
                x1 = ((x1) - (x1 // 10)) - (10 + x1)
                break
                g1 = g1 + 1
            x1 = (3 * x1) + ((-x1) - (-g1))
        else:
            pass
            pass
            return g1
        x1 = ((x1) + (-x1)) - (5 + g1)
    x1 = ((2 + x1) + (5 + g1)) + (x1 % 10)
    return x1


def test_program_9():
    ans = program_9(9)
    assert ans == 200


@hybrid
def program_10(x1, x2):
    pass
    x1 = (2 * x1) + ((x2) * (x2 // 2))
    x1 = ((2 + x2) * (x2 % 10)) * (3 * x2)
    return x1
    return x2


def test_program_10():
    ans = program_10(5, 6)
    assert ans == 864


@hybrid
def program_11(x1):
    pass
    x1 = ((2 + x1) * (x1)) + (x1 // 5)
    x1 = ((5 + x1) + (2 * x1)) - (10 + x1)
    return x1
    return x1


def test_program_11():
    ans = program_11(7)
    assert ans == 123


@hybrid
def program_12(x1, x2):
    x3 = -1
    g1 = -1
    pass
    x3 = (5 + x1) - ((10 + x1) * (5 + x2))
    x2 = ((5 + x1) - (5 + x1)) + (x1 // 10)
    if (x3) * ((10 + x2) - (2 + x2)) < 10:
        pass
        x3 = (2 + x3) - ((-x3) * (5 + x1))
        x3 = (2 + x1) - ((2 * x3) - (-x1))
        g1 = 0
        while g1 < 8:
            pass
            pass
            g1 = g1 + 1
        pass
    pass
    return x3


def test_program_12():
    ans = program_12(4, 1)
    assert ans == 1498


@hybrid
def program_13():
    x1 = -1
    x2 = -1
    x3 = -1
    x4 = -1
    x5 = -1
    g1 = -1
    g2 = -1
    pass
    x1 = 1
    if (-x1) * ((-x1) - (x1)) < 10:
        pass
        x1 = ((2 * x1) - (10 + x1)) + (x1 % 5)
        x2 = ((2 * x1) - (-x1)) - (10 + x1)
        if ((x2) - (5 + x1)) - (x2 * 3) < 10:
            pass
            pass
            if (3 * x2) + ((5 + x1) + (5 + x1)) < 10:
                pass
                pass
                g1 = 0
                while g1 < 9:
                    pass
                    x2 = (3 * x1) + ((-x1) - (x1))
                    g1 = g1 + 1
                    continue
                    g1 = g1 + 1
                x2 = (x2) + ((g1) + (10 + x2))
                x2 = ((x2 + 2) - (x1)) + (2 + g1)
            else:
                pass
                x1 = ((2 * x1) - (5 + g1)) + (x1 + 5)
                return g1
            x1 = ((-g1) - (2 * g1)) - (x2 % 2)
            x3 = ((x1 % 5) - (g1 * 2)) - (2 + g1)
        else:
            pass
            x4 = ((x2) + (10 + x2)) + (x3)
            g2 = 0
            while g2 < 1:
                pass
                pass
                g2 = g2 + 1
            x4 = ((2 + x2) + (2 + x1)) * (2 + x1)
        x4 = ((3 * g2) + (2 * x4)) + (5 + x2)
        x5 = (x3 + 10) * ((5 + g1) - (-x4))
    else:
        pass
        x4 = ((2 * g1) + (3 * x5)) + (-g1)
        x2 = (-x1) - ((10 + g2) * (x4 + 10))
    x2 = (g2 // 2) + ((-x5) + (x5 * 3))
    x2 = ((3 * g2) - (x4 * 2)) * (2 + x2)
    return x5


def test_program_13():
    ans = program_13()
    assert ans == 3114


@hybrid
def program_14(x1, x2, g1):
    x3 = -1
    x4 = -1
    g2 = -1
    pass
    x3 = ((2 * x1) * (3 * g1)) - (x2 + 2)
    x2 = (g1 // 5) + ((x3) * (2 + x1))
    if (g1 % 5) * ((2 * g1) - (2 + x3)) < 10:
        pass
        x1 = ((10 + x3) - (3 * x3)) + (3 * g1)
        x1 = ((3 * x1) * (x3 // 2)) - (g1 + 5)
        if (2 + x2) + ((2 + x3) - (10 + x2)) < 10:
            pass
            x1 = ((x3) - (2 * x1)) * (x3 % 10)
            x2 = ((2 * x2) - (x2)) + (3 * g1)
            if (-g1) - ((2 * x2) - (2 * x3)) < 10:
                pass
                x2 = (3 * x2) * ((10 + g1) + (2 * x1))
                if ((2 * x2) + (x2 // 2)) * (-g1) < 10:
                    pass
                    x3 = (2 + g1) - ((5 + x2) * (x1))
                    x3 = (x1) + ((-x1) - (-x2))
                    return x3
                else:
                    pass
                    pass
                    g2 = 0
                    while g2 < 10:
                        pass
                        pass
                        g2 = g2 + 1
                        continue
                        g2 = g2 + 1
                    x3 = (2 * x2) - ((x3 % 2) * (5 + x1))
                pass
            x1 = (2 + x3) + ((x1) * (3 * x3))
        else:
            pass
            x1 = ((g1) * (x3)) * (g1)
            x2 = ((2 + g1) - (-g2)) * (5 + g1)
        pass
    else:
        pass
        x3 = (5 + x2) * ((x1) * (10 + x3))
        if ((-g2) * (10 + g1)) - (2 * x2) < 10:
            pass
            pass
            while g2 < 6:
                pass
                pass
                g2 = g2 + 1
                continue
                g2 = g2 + 1
            x2 = ((2 + x3) - (x1)) - (5 + x2)
            x4 = ((10 + g1) - (g1 % 10)) - (2 * x1)
        pass
    pass
    return x1


def test_program_14():
    ans = program_14(1, 10, 0)
    assert ans == 87974


@hybrid
def program_15(x1, x2, g1):
    g2 = -1
    pass
    x1 = (2 * x1) * ((5 + g1) * (10 + x1))
    x2 = ((2 * g1) - (-x1)) + (-x2)
    g2 = 0
    while g2 < 6:
        pass
        x2 = (x2) - ((2 + g2) + (g1))
        x2 = ((g2 * 3) * (x1 // 5)) + (2 + x2)
        break
        g2 = g2 + 1
    pass
    return x1


def test_program_15():
    ans = program_15(10, 4, 0)
    assert ans == 2000


@hybrid
def program_16(x1):
    pass
    x1 = ((x1 + 10) - (-x1)) + (10 + x1)
    x1 = ((x1) + (2 * x1)) - (x1 // 5)
    if ((x1 % 5) * (x1 * 2)) * (x1) < 10:
        pass
        x1 = (x1) - ((10 + x1) + (x1 // 10))
        x1 = (2 * x1) * ((x1 + 2) - (x1 % 10))
    x1 = (3 * x1) + ((2 * x1) + (-x1))
    return x1


def test_program_16():
    ans = program_16(3)
    assert ans == 328


@hybrid
def program_17(x1):
    g1 = -1
    g2 = -1
    pass
    x1 = (2 + x1) + ((-x1) + (-x1))
    if (10 + x1) * ((2 + x1) * (5 + x1)) < 10:
        pass
        x1 = ((-x1) * (x1)) + (10 + x1)
        x1 = ((3 * x1) * (-x1)) - (2 + x1)
        g1 = 0
        while g1 < 9:
            pass
            x1 = (2 + g1) * ((g1 % 10) * (x1 + 2))
            x1 = (x1) - ((g1 // 10) * (2 + x1))
            g2 = 0
            while g2 < 9:
                pass
                x1 = ((2 + x1) + (g2)) * (g1)
                x1 = ((3 * g2) + (g1)) + (3 * x1)
                break
                g2 = g2 + 1
            x1 = (10 + g2) * ((-g2) - (3 * g1))
            x1 = ((-x1) * (2 + x1)) + (x1 // 10)
            g1 = g1 + 1
        x1 = (2 * g1) - ((-g1) * (2 + g2))
        x1 = ((2 * g1) + (2 * g2)) * (5 + x1)
    x1 = (x1) * ((x1) + (g1))
    return x1


def test_program_17():
    ans = program_17(7)
    assert ans == 551286


@hybrid
def program_18(x1):
    x2 = -1
    g1 = -1
    pass
    x1 = (x1 // 5) - ((2 * x1) + (10 + x1))
    if ((x1 % 5) * (5 + x1)) - (3 * x1) < 10:
        pass
        x1 = ((-x1) * (x1 // 5)) * (x1 * 3)
        x2 = ((2 + x1) + (x1)) - (3 * x1)
        return x2
    else:
        pass
        x1 = ((x1 + 5) + (x1)) + (x1 % 5)
        if ((x1 // 5) - (2 + x1)) + (2 + x1) < 10:
            pass
            pass
            g1 = 0
            while g1 < 2:
                pass
                x1 = ((10 + x1) + (2 * x1)) * (-g1)
                break
                g1 = g1 + 1
            x2 = ((x1 + 5) - (10 + x2)) * (5 + x1)
        else:
            pass
            x2 = (5 + x1) + ((5 + x1) * (g1))
            x2 = ((5 + x2) + (x1 % 2)) + (x1 % 5)
            return x1
        x2 = ((-g1) - (x1 + 2)) * (x2)
        x2 = (-x2) - ((g1 % 2) + (x1 // 10))
    x1 = ((x1) - (3 * x2)) + (g1)
    return x1


def test_program_18():
    ans = program_18(9)
    assert ans == 120


@hybrid
def program_19(g1):
    x1 = -1
    x2 = -1
    pass
    x1 = ((2 + g1) + (2 + g1)) + (3 * g1)
    while g1 < 4:
        pass
        x1 = (x1 // 2) + ((2 + g1) * (g1 + 5))
        x2 = ((x1) * (x1)) * (x1)
        return x2
        g1 = g1 + 1
    pass
    return x2


def test_program_19():
    ans = program_19(0)
    assert ans == 1728


@hybrid
def program_20(x1, g1, g2):
    x2 = -1
    x3 = -1
    pass
    x1 = (2 * g1) + ((x1 // 10) - (g2))
    while g1 < 7:
        pass
        x1 = ((10 + g2) - (2 + g2)) * (x1 // 2)
        while g1 < 7:
            pass
            x2 = (x1) * ((2 + x1) + (2 * x1))
            while g2 < 2:
                pass
                x1 = (x1 + 2) - ((3 * g2) * (5 + x1))
                x3 = ((2 + x2) * (10 + x2)) + (3 * x1)
                g2 = g2 + 1
                continue
                g2 = g2 + 1
            x2 = ((x1 * 3) - (-x1)) + (3 * x2)
            x3 = (g1 + 5) + ((2 + x1) * (5 + x1))
            g1 = g1 + 1
        x2 = ((3 * g2) + (x3)) + (5 + x3)
        g1 = g1 + 1
    x3 = (10 + x2) - ((5 + g1) * (2 + g1))
    return x2


def test_program_20():
    ans = program_20(5, 0, 0)
    assert ans == 393


@hybrid
def program_21(x1):
    x2 = -1
    g1 = -1
    pass
    pass
    g1 = 0
    while g1 < 2:
        pass
        x1 = ((10 + x1) * (2 + x1)) * (x1)
        x1 = (2 + g1) * ((-g1) + (5 + x1))
        break
        g1 = g1 + 1
    x2 = (5 + x1) - ((2 + x1) + (5 + g1))
    return x1


def test_program_21():
    ans = program_21(8)
    assert ans == 2890


@hybrid
def program_22(g1):
    x1 = -1
    pass
    x1 = ((g1) * (2 + g1)) + (3 * g1)
    x1 = (-x1) - ((g1 % 10) * (g1 % 2))
    while g1 < 8:
        pass
        pass
        break
        g1 = g1 + 1
    x1 = (5 + g1) * ((10 + g1) - (x1))
    x1 = (x1) * ((2 * x1) + (10 + g1))
    return x1


def test_program_22():
    ans = program_22(0)
    assert ans == 5500


@hybrid
def program_23(x1, x2, g1, g2):
    x3 = -1
    g3 = -1
    pass
    x1 = (2 + g2) + ((2 * g1) + (2 * x2))
    x1 = ((5 + g2) + (2 + g1)) + (2 * x1)
    if ((2 + x2) - (2 * x1)) - (2 * g2) < 10:
        pass
        x3 = ((5 + g1) * (5 + x1)) + (g1)
        g3 = 0
        while g3 < 3:
            pass
            pass
            break
            g3 = g3 + 1
        x3 = (x3) - ((-g3) + (g1 % 5))
        x2 = (-x2) - ((10 + x2) * (g2))
    else:
        pass
        x1 = (5 + x3) * ((x1 // 2) + (5 + x1))
        x1 = ((-g2) - (2 * g2)) + (-x3)
        while g1 < 1:
            pass
            pass
            return x3
            g1 = g1 + 1
        x2 = (-x1) - ((2 * g1) - (x3 % 5))
        x1 = (3 * x3) * ((3 * g3) - (x2))
    x1 = ((3 * g1) - (x1 % 5)) + (g1)
    return x3


def test_program_23():
    ans = program_23(8, 9, 0, 0)
    assert ans == 260


@hybrid
def program_24(x1, g1):
    g2 = -1
    pass
    pass
    g2 = 0
    while g2 < 6:
        pass
        x1 = ((2 * g2) + (g2 + 10)) + (g2)
        if ((g1 // 2) * (10 + g2)) + (-x1) < 10:
            pass
            x1 = ((2 * g2) - (-g1)) * (5 + g2)
            g2 = g2 + 1
            continue
        pass
        g2 = g2 + 1
    pass
    return x1


def test_program_24():
    ans = program_24(3, 0)
    assert ans == 100


@hybrid
def program_25():
    x1 = -1
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x1 = 0
    g1 = 0
    while g1 < 6:
        pass
        x1 = (g1 * 2) * ((2 + g1) + (g1))
        x2 = (2 + x1) - ((2 + g1) + (-x1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    x2 = (3 * x1) * ((2 * x1) + (g1 % 5))
    x3 = ((5 + x2) + (10 + x2)) * (2 * x2)
    return x1


def test_program_25():
    ans = program_25()
    assert ans == 120


@hybrid
def program_26(x1):
    pass
    x1 = ((-x1) + (10 + x1)) + (5 + x1)
    x1 = (3 * x1) * ((x1) * (3 * x1))
    return x1
    return x1


def test_program_26():
    ans = program_26(9)
    assert ans == 124416


@hybrid
def program_27(x1, x2, g1, g2):
    x3 = -1
    pass
    pass
    while g1 < 8:
        pass
        x1 = ((10 + x1) * (5 + g1)) + (10 + x2)
        x1 = (x1) + ((-x1) + (2 + g1))
        while g2 < 6:
            pass
            x1 = (5 + g1) * ((10 + x1) * (x2 + 2))
            return x1
            g2 = g2 + 1
        pass
        g1 = g1 + 1
    x3 = ((x2 // 10) * (5 + x2)) * (10 + x1)
    return g1


def test_program_27():
    ans = program_27(2, 7, 0, 0)
    assert ans == 540


@hybrid
def program_28():
    x1 = -1
    g1 = -1
    g2 = -1
    pass
    x1 = 3
    if (x1 % 5) * ((x1 // 5) * (5 + x1)) < 10:
        pass
        x1 = (5 + x1) - ((10 + x1) * (3 * x1))
        g1 = 0
        while g1 < 1:
            pass
            x1 = ((3 * x1) + (2 * g1)) * (x1 // 10)
            return x1
            g1 = g1 + 1
        x1 = (10 + g1) - ((10 + g1) + (x1 % 10))
    else:
        pass
        pass
        if (g1 // 5) + ((5 + x1) * (g1 * 2)) < 10:
            pass
            x1 = (2 * x1) + ((x1) * (2 + g1))
            x1 = ((x1 + 2) + (-g1)) + (2 + x1)
            return g1
        else:
            pass
            x1 = ((2 * x1) + (5 + g1)) + (5 + x1)
            g2 = 0
            while g2 < 9:
                pass
                x1 = (5 + x1) * ((2 * g2) * (g2))
                break
                g2 = g2 + 1
            x1 = ((3 * x1) + (5 + x1)) * (10 + x1)
            x1 = (g2 % 5) * ((2 * g2) + (10 + x1))
        pass
    pass
    return x1


def test_program_28():
    ans = program_28()
    assert ans == 3270


@hybrid
def program_29(x1, x2):
    g1 = -1
    g2 = -1
    g3 = -1
    pass
    x2 = (5 + x1) - ((x1 // 10) + (2 * x2))
    x2 = ((x2 // 5) - (2 * x1)) * (3 * x1)
    g1 = 0
    while g1 < 1:
        pass
        pass
        g2 = 0
        while g2 < 10:
            pass
            pass
            g3 = 0
            while g3 < 4:
                pass
                x2 = (5 + g2) + ((3 * g2) - (g2 + 5))
                g3 = g3 + 1
                continue
                g3 = g3 + 1
            x2 = ((g2 % 10) - (3 * x2)) + (2 + g2)
            g2 = g2 + 1
        x2 = (5 + x1) + ((g3 * 2) + (5 + x1))
        g1 = g1 + 1
    x1 = ((g3 + 10) + (2 + x1)) * (5 + g3)
    return x1


def test_program_29():
    ans = program_29(9, 2)
    assert ans == 225


@hybrid
def program_30(x1, g1):
    x2 = -1
    x3 = -1
    x4 = -1
    pass
    pass
    if ((5 + x1) - (-g1)) + (2 * x1) < 10:
        pass
        x1 = (2 * x1) - ((g1 // 10) + (10 + x1))
        if ((2 + g1) * (-x1)) + (2 + x1) < 10:
            pass
            pass
            if (5 + x1) - ((-x1) + (2 * x1)) < 10:
                pass
                pass
                return x1
            x1 = (5 + x1) * ((x1 + 2) + (2 + g1))
        else:
            pass
            pass
            while g1 < 3:
                pass
                x1 = (-g1) + ((-x1) * (2 * x1))
                g1 = g1 + 1
                continue
                g1 = g1 + 1
            x2 = ((g1) - (x1 * 2)) - (g1 * 3)
        x3 = (x2 + 2) + ((5 + g1) * (-x1))
        x3 = ((3 * x2) * (x2)) - (10 + x3)
    else:
        pass
        x2 = (2 * g1) - ((x3 % 10) * (10 + x1))
        x4 = ((x3 * 2) * (5 + x3)) + (10 + x2)
        if ((3 * x4) * (x4 // 2)) * (x3 + 2) < 10:
            pass
            pass
            return x3
        else:
            pass
            x3 = ((2 * x2) * (10 + x2)) * (2 + x4)
            x2 = ((2 * x4) * (2 * x3)) * (g1 + 2)
        x3 = ((x1 // 10) - (2 + x1)) * (10 + x4)
    x1 = (3 * x1) - ((3 * x3) - (x1))
    return x1


def test_program_30():
    ans = program_30(3, 0)
    assert ans == 387


@hybrid
def program_31(x1, x2, g1):
    x3 = -1
    x4 = -1
    pass
    x2 = ((5 + x2) + (x1 // 10)) - (3 * x2)
    x3 = (3 * x2) * ((x1 + 10) + (-g1))
    while g1 < 9:
        pass
        pass
        if (2 + g1) - ((10 + x2) - (2 * x3)) < 10:
            pass
            x4 = (10 + x3) * ((-x1) + (10 + g1))
            break
        else:
            pass
            pass
            g1 = g1 + 1
            continue
        x3 = ((2 + x1) + (x3)) * (-x3)
        x4 = ((x1 % 10) + (x2)) * (2 + x3)
        g1 = g1 + 1
    x3 = (-x4) + ((3 * x1) * (10 + x4))
    return x3


def test_program_31():
    ans = program_31(7, 1, 0)
    assert ans == 190


@hybrid
def program_32():
    x1 = -1
    x2 = -1
    g1 = -1
    pass
    pass
    if 2 < 10:
        pass
        x1 = 2
        x1 = ((2 + x1) - (x1)) * (x1 + 5)
        g1 = 0
        while g1 < 5:
            pass
            pass
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        x1 = (2 * x1) - ((3 * x1) + (3 * x1))
    else:
        pass
        x1 = (2 + x1) - ((x1 % 10) - (10 + g1))
        x1 = (10 + x1) + ((2 + x1) * (g1 % 2))
        while g1 < 8:
            pass
            x1 = (-g1) + ((5 + g1) * (x1 // 2))
            break
            g1 = g1 + 1
        x2 = (g1 // 10) * ((-g1) * (5 + g1))
    x1 = ((5 + x1) * (-g1)) + (g1)
    return x1


def test_program_32():
    ans = program_32()
    assert ans == 260


@hybrid
def program_33(x1, x2):
    pass
    x2 = ((10 + x2) * (3 * x2)) * (10 + x2)
    return x2
    return x1


def test_program_33():
    ans = program_33(7, 1)
    assert ans == 363


@hybrid
def program_34(x1):
    g1 = -1
    pass
    x1 = ((3 * x1) * (x1)) * (x1 + 2)
    g1 = 0
    while g1 < 6:
        pass
        pass
        while g1 < 5:
            pass
            x1 = ((2 + g1) + (3 * g1)) * (x1)
            x1 = (g1 * 2) - ((10 + x1) * (10 + g1))
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        pass
        g1 = g1 + 1
    x1 = ((5 + g1) * (10 + g1)) - (3 * g1)
    x1 = ((x1) * (3 * g1)) + (3 * x1)
    return x1


def test_program_34():
    ans = program_34(1)
    assert ans == 3318


@hybrid
def program_35(x1, x2):
    x3 = -1
    x4 = -1
    g1 = -1
    g2 = -1
    g3 = -1
    pass
    x1 = (3 * x1) * ((3 * x2) + (x2 * 2))
    x2 = ((x1 % 5) * (x1 // 5)) + (2 + x1)
    g1 = 0
    while g1 < 9:
        pass
        pass
        while g1 < 7:
            pass
            x1 = ((5 + x1) * (2 * x2)) * (x1)
            while g1 < 3:
                pass
                x1 = (-g1) - ((x2) - (2 + g1))
                x2 = ((3 * g1) * (3 * x1)) + (5 + g1)
                g2 = 0
                while g2 < 1:
                    pass
                    x1 = (10 + g1) + ((3 * g2) * (10 + x1))
                    g3 = 0
                    while g3 < 1:
                        pass
                        x3 = ((2 * x2) * (3 * g3)) - (2 * x2)
                        while g1 < 4:
                            pass
                            x2 = (g3) - ((-x1) - (5 + x1))
                            x3 = ((10 + g1) + (2 + x1)) + (-x2)
                            break
                            g1 = g1 + 1
                        x3 = (x2 + 5) * ((x1) - (-g1))
                        g3 = g3 + 1
                    x1 = (2 + x2) - ((5 + x1) * (3 * x1))
                    x2 = (5 + x1) + ((3 * x3) + (x1 * 2))
                    g2 = g2 + 1
                x2 = (5 + g2) + ((3 * g1) - (x1 // 5))
                x3 = (g3 % 10) + ((x2 * 3) + (2 * x3))
                g1 = g1 + 1
            x1 = ((10 + g1) - (10 + g2)) + (2 + g3)
            g1 = g1 + 1
        x4 = (x3 * 2) * ((-g1) + (x1 % 5))
        x1 = ((-x4) - (5 + g3)) + (x2 * 3)
        g1 = g1 + 1
    pass
    return x3


def test_program_35():
    ans = program_35(0, 0)
    assert ans == 1337


@hybrid
def program_36(x1, x2, g1):
    x3 = -1
    x4 = -1
    x5 = -1
    g2 = -1
    pass
    x3 = ((x2 % 5) - (g1)) + (2 + x1)
    if ((x2 * 2) * (x3)) + (-x1) < 10:
        pass
        pass
    else:
        pass
        x3 = ((x3) * (2 + x2)) - (5 + x2)
        x3 = (3 * x2) + ((2 + x3) + (2 * g1))
        g2 = 0
        while g2 < 10:
            pass
            x4 = ((x2 % 10) - (2 * x2)) * (x2 + 2)
            if ((-g2) - (10 + g2)) - (2 + x2) < 10:
                pass
                x1 = ((2 + x3) - (3 * g2)) * (-x2)
                g2 = g2 + 1
                continue
            else:
                pass
                x4 = ((2 + g2) + (10 + g1)) - (2 * x2)
                break
            pass
            g2 = g2 + 1
        pass
    x5 = (3 * x4) * ((g1) + (x4 // 5))
    return x3


def test_program_36():
    ans = program_36(4, 9, 0)
    assert ans == 125


@hybrid
def program_37(x1, x2, g1):
    pass
    x2 = ((10 + x2) * (5 + x2)) + (5 + g1)
    return x2
    return x1


def test_program_37():
    ans = program_37(10, 5, 0)
    assert ans == 155


@hybrid
def program_38(g1, g2):
    x1 = -1
    x2 = -1
    x3 = -1
    x4 = -1
    g3 = -1
    pass
    x1 = (3 * g1) - ((2 * g1) * (3 * g2))
    x2 = ((g1 % 2) - (-g1)) * (-g1)
    if (g2 % 5) + ((2 + g1) - (2 + x2)) < 10:
        pass
        x3 = (10 + x1) * ((-x1) * (2 + g2))
        g3 = 0
        while g3 < 8:
            pass
            x3 = (10 + x1) - ((2 + x2) - (2 + g3))
            x3 = (-x2) + ((10 + x2) + (g1))
            break
            g3 = g3 + 1
        x1 = ((g2) - (g2 % 10)) - (2 * g3)
        x1 = ((3 * g1) + (5 + x3)) * (x2 % 5)
    else:
        pass
        x1 = ((g3) * (x1)) - (x3 * 3)
        x3 = ((10 + x3) - (x2 // 10)) * (2 + g3)
    x4 = ((10 + g1) * (x3)) - (2 * x2)
    return x4


def test_program_38():
    ans = program_38(0, 0)
    assert ans == 100


@hybrid
def program_39(x1, x2):
    pass
    x1 = (x1 + 10) * ((3 * x1) * (x1))
    x2 = ((-x2) * (-x2)) + (x2 + 2)
    return x1
    return x2


def test_program_39():
    ans = program_39(5, 6)
    assert ans == 1125


@hybrid
def program_40(x1, g1, g2):
    x2 = -1
    x3 = -1
    pass
    x2 = (g2 % 2) * ((10 + g2) - (x1))
    x1 = (2 * g2) + ((10 + x1) + (x2))
    while g1 < 6:
        pass
        x2 = ((2 * g2) - (2 + g2)) + (5 + x1)
        x3 = ((x1) * (x1 + 10)) - (3 * x2)
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    pass
    return x3


def test_program_40():
    ans = program_40(5, 0, 0)
    assert ans == 321


@hybrid
def program_41():
    x1 = -1
    pass
    x1 = 1
    if ((2 + x1) + (-x1)) + (-x1) < 10:
        pass
        x1 = ((5 + x1) + (2 + x1)) - (x1 % 10)
    else:
        pass
        x1 = ((2 + x1) - (-x1)) - (2 * x1)
        return x1
    x1 = ((3 * x1) - (5 + x1)) + (x1)
    x1 = (2 * x1) * ((2 + x1) + (2 + x1))
    return x1


def test_program_41():
    ans = program_41()
    assert ans == 1596


@hybrid
def program_42():
    x1 = -1
    g1 = -1
    pass
    x1 = 2
    x1 = ((3 * x1) + (10 + x1)) * (2 + x1)
    g1 = 0
    while g1 < 6:
        pass
        x1 = (x1 * 2) * ((x1 + 5) - (5 + g1))
        x1 = ((x1) + (3 * x1)) + (g1)
        break
        g1 = g1 + 1
    pass
    return x1


def test_program_42():
    ans = program_42()
    assert ans == 41472


@hybrid
def program_43(x1, x2):
    x3 = -1
    x4 = -1
    g1 = -1
    g2 = -1
    pass
    x1 = ((x1) + (2 * x2)) * (2 + x2)
    x2 = ((x2 + 2) + (10 + x1)) * (10 + x1)
    if (3 * x2) * ((2 + x1) + (5 + x2)) < 10:
        pass
        x3 = (x1 + 10) * ((-x1) + (-x2))
        if ((2 + x1) * (-x1)) - (x1) < 10:
            pass
            x2 = (3 * x3) * ((5 + x3) + (x1))
            x2 = ((2 * x1) * (x1)) + (10 + x2)
            g1 = 0
            while g1 < 9:
                pass
                pass
                break
                g1 = g1 + 1
            x2 = (5 + g1) + ((3 * x2) * (3 * g1))
        pass
    else:
        pass
        x3 = ((x3 + 5) - (10 + g1)) + (10 + x2)
        x4 = ((2 + x2) + (x1 // 10)) * (5 + g1)
        if (x4 * 3) * ((3 * x1) - (g1 + 10)) < 10:
            pass
            pass
            g2 = 0
            while g2 < 8:
                pass
                pass
                while g1 < 5:
                    pass
                    x4 = ((x4 // 2) + (x2)) - (-g2)
                    g1 = g1 + 1
                    continue
                    g1 = g1 + 1
                pass
                g2 = g2 + 1
            pass
        x4 = ((x3 // 2) - (2 * g2)) - (g2)
    x4 = ((5 + g2) + (x4)) + (2 + x1)
    x1 = (-x4) + ((x3 % 10) - (x3 * 2))
    return x2


def test_program_43():
    ans = program_43(1, 10)
    assert ans == 71788


@hybrid
def program_44(g1, g2):
    x1 = -1
    x2 = -1
    pass
    x1 = (g2) * ((g2) * (5 + g1))
    x2 = ((10 + g1) * (10 + g2)) * (10 + g1)
    while g2 < 6:
        pass
        x1 = (3 * x2) - ((-g1) - (g1 // 2))
        x2 = ((x1 % 10) - (g1 * 3)) - (-x1)
        break
        g2 = g2 + 1
    x2 = ((10 + g1) * (x1)) - (10 + x1)
    return x2


def test_program_44():
    ans = program_44(0, 0)
    assert ans == 26990


@hybrid
def program_45(g1):
    x1 = -1
    x2 = -1
    x3 = -1
    pass
    x1 = (10 + g1) + ((2 * g1) - (5 + g1))
    while g1 < 3:
        pass
        x2 = ((x1) * (x1)) - (2 * x1)
        while g1 < 10:
            pass
            x1 = (2 * x2) * ((10 + x2) * (x2))
            x3 = (2 * x1) - ((x1 // 5) * (5 + x2))
            if (2 * x2) - ((g1 % 5) - (3 * x1)) < 10:
                pass
                pass
                g1 = g1 + 1
                continue
            pass
            g1 = g1 + 1
        x3 = ((2 + x1) - (5 + x1)) * (2 + x3)
        g1 = g1 + 1
    pass
    return x1


def test_program_45():
    ans = program_45(0)
    assert ans == 11250


@hybrid
def program_46(x1):
    pass
    x1 = ((x1) * (5 + x1)) - (2 * x1)
    x1 = ((x1) * (2 * x1)) + (x1)
    return x1


def test_program_46():
    ans = program_46(6)
    assert ans == 5886


@hybrid
def program_47(x1):
    pass
    x1 = ((x1) * (2 * x1)) * (x1 % 10)
    return x1
    return x1


def test_program_47():
    ans = program_47(8)
    assert ans == 1024


@hybrid
def program_48(g1):
    x1 = -1
    x2 = -1
    pass
    x1 = ((g1 % 5) + (2 + g1)) * (2 + g1)
    x1 = (-x1) * ((x1 % 5) - (3 * g1))
    while g1 < 7:
        pass
        x2 = (2 + x1) * ((5 + x1) - (10 + g1))
        while g1 < 5:
            pass
            pass
            return x2
            g1 = g1 + 1
        x2 = (2 * x1) - ((g1 // 5) - (x1))
        g1 = g1 + 1
    x1 = ((2 * g1) * (3 * x2)) + (2 * x2)
    return g1


def test_program_48():
    ans = program_48(0)
    assert ans == 294


@hybrid
def program_49(g1):
    x1 = -1
    g2 = -1
    pass
    x1 = ((10 + g1) + (10 + g1)) + (2 + g1)
    x1 = (2 + g1) * ((3 * x1) + (x1 + 5))
    g2 = 0
    while g2 < 2:
        pass
        pass
        if ((10 + g2) * (-x1)) - (x1) < 10:
            pass
            x1 = ((3 * g2) - (3 * x1)) - (g2)
            x1 = (g2) - ((2 + g2) - (2 * x1))
            break
        pass
        g2 = g2 + 1
    x1 = ((2 + g1) - (g1 * 3)) - (x1 + 5)
    return x1


def test_program_49():
    ans = program_49(0)
    assert ans == 1115


@hybrid
def program_50(x1):
    x2 = -1
    g1 = -1
    pass
    x2 = ((3 * x1) + (5 + x1)) * (10 + x1)
    x2 = ((3 * x2) - (x2)) + (x2)
    g1 = 0
    while g1 < 9:
        pass
        pass
        break
        g1 = g1 + 1
    pass
    return x2


def test_program_50():
    ans = program_50(1)
    assert ans == 297


@hybrid
def program_51():
    x1 = -1
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x1 = 1
    g1 = 0
    while g1 < 7:
        pass
        x2 = ((5 + x1) - (x1)) * (g1 + 2)
        x2 = (3 * x2) - ((3 * x1) - (5 + x1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    x3 = ((2 + x2) * (2 * x2)) + (5 + g1)
    x1 = (10 + g1) + ((2 * x2) + (2 * x2))
    return x1


def test_program_51():
    ans = program_51()
    assert ans == 509


@hybrid
def program_52(x1, x2, g1, g2):
    x3 = -1
    g3 = -1
    pass
    x1 = (3 * g2) - ((2 + g1) + (-g2))
    if ((g2) - (3 * g2)) - (g1 * 2) < 10:
        pass
        pass
        g3 = 0
        while g3 < 8:
            pass
            x2 = ((3 * x1) - (10 + g2)) + (g3)
            x1 = (5 + g2) - ((x2 * 2) * (g3 + 5))
            break
            g3 = g3 + 1
        x2 = ((2 * g1) * (x2 // 2)) - (g2 // 5)
        x3 = (5 + x1) * ((2 + x1) - (x1 * 3))
    else:
        pass
        pass
    x2 = (10 + g2) * ((-g2) * (x3))
    x2 = (10 + x2) * ((2 * x2) * (5 + g3))
    return x1


def test_program_52():
    ans = program_52(10, 10, 0, 0)
    assert ans == 165


@hybrid
def program_53():
    x1 = -1
    x2 = -1
    x3 = -1
    g1 = -1
    g2 = -1
    pass
    pass
    g1 = 0
    while g1 < 10:
        pass
        pass
        g2 = 0
        while g2 < 7:
            pass
            x1 = ((g2 + 10) * (10 + g2)) + (3 * g1)
            x2 = (2 + g2) * ((5 + g1) - (x1 % 5))
            g2 = g2 + 1
            continue
            g2 = g2 + 1
        x2 = ((2 * x2) - (10 + x2)) + (3 * g1)
        g1 = g1 + 1
    x2 = ((2 + x1) * (2 * g2)) - (10 + g2)
    x3 = ((-x1) * (3 * x2)) + (2 * x1)
    return x1


def test_program_53():
    ans = program_53()
    assert ans == 283


@hybrid
def program_54(x1):
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x2 = (2 + x1) * ((-x1) + (2 * x1))
    x3 = ((2 + x2) - (10 + x2)) + (3 * x2)
    g1 = 0
    while g1 < 6:
        pass
        x1 = ((g1) - (x3)) - (x2 // 2)
        break
        g1 = g1 + 1
    pass
    return x3


def test_program_54():
    ans = program_54(6)
    assert ans == 136


@hybrid
def program_55(g1, g2):
    x1 = -1
    x2 = -1
    g3 = -1
    pass
    x1 = (-g2) - ((3 * g2) * (-g2))
    x2 = ((g2 % 5) + (5 + x1)) * (x1 // 5)
    while g1 < 2:
        pass
        x2 = (2 * g1) - ((2 * g2) - (g1 % 5))
        g3 = 0
        while g3 < 3:
            pass
            x1 = ((-g1) + (x2 % 5)) * (10 + x1)
            g3 = g3 + 1
            continue
            g3 = g3 + 1
        x2 = ((x2 % 10) + (5 + g2)) * (x1 + 2)
        x1 = ((5 + g3) + (2 * g1)) + (-x2)
        g1 = g1 + 1
    x1 = (-g3) - ((g3) - (g1))
    x1 = (-g1) - ((g2 * 2) + (5 + g1))
    return x2


def test_program_55():
    ans = program_55(0, 0)
    assert ans == 1008


@hybrid
def program_56():
    x1 = -1
    g1 = -1
    pass
    pass
    if 3 < 10:
        pass
        pass
        g1 = 0
        while g1 < 2:
            pass
            x1 = ((3 * g1) + (3 * g1)) + (g1)
            x1 = (5 + x1) - ((3 * x1) + (-g1))
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        x1 = (5 + g1) + ((2 + x1) - (3 * g1))
        x1 = ((2 * g1) + (g1 + 10)) + (2 + x1)
    x1 = (-x1) - ((2 * x1) * (5 + x1))
    x1 = ((x1 % 5) + (x1)) * (3 * x1)
    return x1


def test_program_56():
    ans = program_56()
    assert ans == 695526


@hybrid
def program_57(x1):
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x2 = ((3 * x1) * (2 * x1)) + (-x1)
    x2 = (x2 % 2) * ((x2) + (x1 // 2))
    g1 = 0
    while g1 < 9:
        pass
        x1 = ((2 * x1) * (-g1)) + (5 + x1)
        x3 = (5 + x1) * ((3 * g1) - (x1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    pass
    return x2


def test_program_57():
    ans = program_57(7)
    assert ans == 290


@hybrid
def program_58(x1, x2):
    pass
    x1 = (2 * x1) + ((x1) - (x1 % 10))
    x1 = ((2 * x1) - (x2)) * (x1)
    return x1
    return x1


def test_program_58():
    ans = program_58(8, 10)
    assert ans == 352


@hybrid
def program_59(g1):
    x1 = -1
    g2 = -1
    pass
    x1 = (g1 // 5) - ((g1) - (g1 // 10))
    g2 = 0
    while g2 < 10:
        pass
        x1 = (2 * g1) * ((2 + x1) - (5 + g1))
        if ((-g2) + (g2 // 10)) - (3 * x1) < 10:
            pass
            x1 = ((10 + x1) * (2 * g2)) - (3 * g2)
            x1 = (g2 + 10) - ((3 * g2) + (g2 + 5))
            while g1 < 5:
                pass
                pass
                if (g1 // 5) + ((x1 * 2) + (g2 + 5)) < 10:
                    pass
                    pass
                    g1 = g1 + 1
                    continue
                else:
                    pass
                    pass
                    g1 = g1 + 1
                    continue
                x1 = ((x1 * 3) * (10 + x1)) * (2 * g2)
                g1 = g1 + 1
            x1 = (2 * g2) - ((10 + g2) - (2 * g1))
            x1 = ((g1) - (g2 + 5)) - (x1)
        else:
            pass
            x1 = ((-g2) + (5 + g1)) * (g2 % 10)
            x1 = (2 * x1) * ((10 + x1) * (x1 % 10))
            return x1
        pass
        g2 = g2 + 1
    x1 = (2 * g1) * ((x1) - (3 * x1))
    x1 = (3 * x1) + ((g1 // 5) * (5 + g1))
    return x1


def test_program_59():
    ans = program_59(0)
    assert ans == 3078


@hybrid
def program_60(g1, g2):
    x1 = -1
    x2 = -1
    x3 = -1
    x4 = -1
    g3 = -1
    g4 = -1
    pass
    x1 = ((2 + g2) * (g2)) * (10 + g2)
    g3 = 0
    while g3 < 2:
        pass
        x1 = ((5 + x1) - (-g3)) * (3 * g3)
        g4 = 0
        while g4 < 7:
            pass
            x2 = ((-x1) * (-g4)) * (g2)
            x3 = ((g4 // 2) + (g1 // 10)) * (3 * g1)
            while g1 < 7:
                pass
                x3 = (2 * x3) + ((2 + x2) * (3 * x3))
                x1 = (x1 // 5) * ((5 + g1) - (2 * x1))
                while g3 < 1:
                    pass
                    x2 = (g1) + ((5 + g3) + (5 + g1))
                    x2 = ((-g1) + (10 + g2)) - (g2 % 10)
                    break
                    g3 = g3 + 1
                pass
                g1 = g1 + 1
            x1 = ((g2 + 5) * (5 + x3)) - (g4 * 3)
            g4 = g4 + 1
        x2 = (-g2) + ((g3) - (x3 * 3))
        x3 = ((2 + x1) * (2 * g2)) * (g4 + 5)
        g3 = g3 + 1
    x4 = ((g2) - (2 + g3)) - (-g2)
    return x1


def test_program_60():
    ans = program_60(0, 0)
    assert ans == 322


@hybrid
def program_61(x1, x2):
    x3 = -1
    g1 = -1
    pass
    x3 = ((5 + x1) - (x1 * 3)) + (x2)
    g1 = 0
    while g1 < 8:
        pass
        x2 = (10 + x2) + ((g1 // 5) - (-x1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    pass
    return x2


def test_program_61():
    ans = program_61(10, 2)
    assert ans == 165


@hybrid
def program_62(x1):
    x2 = -1
    g1 = -1
    pass
    x1 = (x1) - ((x1) * (3 * x1))
    x1 = (5 + x1) + ((3 * x1) * (-x1))
    g1 = 0
    while g1 < 8:
        pass
        x1 = (-x1) - ((-g1) + (g1 + 5))
        return x1
        g1 = g1 + 1
    x2 = (g1 % 2) * ((3 * x1) + (-x1))
    return x1


def test_program_62():
    ans = program_62(9)
    assert ans == 164492


@hybrid
def program_63(x1, x2, g1):
    x3 = -1
    x4 = -1
    g2 = -1
    pass
    x2 = (3 * g1) * ((-g1) * (x1 * 2))
    x1 = (2 + x2) * ((x1 % 5) - (-x1))
    g2 = 0
    while g2 < 10:
        pass
        x3 = ((2 + x1) * (5 + g1)) - (2 + x1)
        x1 = ((3 * x1) + (2 * g2)) + (x3 // 10)
        g2 = g2 + 1
        continue
        g2 = g2 + 1
    x2 = (10 + x3) * ((x3) * (-x1))
    x4 = ((5 + x3) + (3 * x3)) + (g1)
    return x3


def test_program_63():
    ans = program_63(0, 10, 0)
    assert ans == 87020


@hybrid
def program_64(x1, x2, g1, g2):
    pass
    x2 = ((5 + x2) + (g1)) * (10 + g2)
    x1 = ((x2 // 2) * (2 + g2)) - (5 + x1)
    return x1
    return g2


def test_program_64():
    ans = program_64(2, 10, 0, 0)
    assert ans == 143


@hybrid
def program_65(x1, x2):
    x3 = -1
    g1 = -1
    pass
    x2 = ((3 * x2) + (-x1)) * (2 + x1)
    x3 = ((2 + x1) + (3 * x2)) * (3 * x1)
    g1 = 0
    while g1 < 6:
        pass
        pass
        while g1 < 5:
            pass
            pass
            if (3 * x3) * ((x3) - (5 + x3)) < 10:
                pass
                pass
                while g1 < 10:
                    pass
                    pass
                    break
                    g1 = g1 + 1
                pass
            pass
            g1 = g1 + 1
        x1 = ((3 * g1) + (5 + x1)) + (-g1)
        g1 = g1 + 1
    pass
    return x3


def test_program_65():
    ans = program_65(6, 8)
    assert ans == 7920


@hybrid
def program_66(x1):
    pass
    x1 = (x1 * 3) * ((x1 * 2) + (x1 % 10))
    return x1
    return x1


def test_program_66():
    ans = program_66(8)
    assert ans == 576


@hybrid
def program_67(x1, x2, g1):
    g2 = -1
    pass
    pass
    g2 = 0
    while g2 < 8:
        pass
        x2 = (5 + x1) * ((2 * x2) * (2 * x1))
        return x2
        g2 = g2 + 1
    x1 = (5 + g2) - ((10 + x2) + (5 + g2))
    return x1


def test_program_67():
    ans = program_67(10, 6, 0)
    assert ans == 3600


@hybrid
def program_68(x1):
    pass
    x1 = (5 + x1) * ((2 + x1) * (3 * x1))
    return x1


def test_program_68():
    ans = program_68(9)
    assert ans == 4158


@hybrid
def program_69(x1, x2):
    x3 = -1
    x4 = -1
    x5 = -1
    g1 = -1
    pass
    x3 = ((x1 + 10) * (10 + x1)) - (10 + x1)
    x1 = ((5 + x2) * (5 + x3)) * (10 + x3)
    g1 = 0
    while g1 < 9:
        pass
        x4 = ((-x2) + (3 * x3)) - (g1)
        x2 = (2 * x3) * ((x1) + (2 + g1))
        while g1 < 8:
            pass
            x5 = (2 * g1) * ((-g1) + (-x3))
            break
            g1 = g1 + 1
        pass
        g1 = g1 + 1
    pass
    return x3


def test_program_69():
    ans = program_69(6, 5)
    assert ans == 240


@hybrid
def program_70(x1, x2, g1):
    pass
    x1 = (x1) + ((x1 + 5) + (10 + x1))
    while g1 < 1:
        pass
        x2 = (5 + g1) + ((10 + x1) - (5 + g1))
        x1 = ((-x1) - (g1)) + (10 + x2)
        if (x2 % 2) * ((-x2) - (-x2)) < 10:
            pass
            x2 = (2 + g1) - ((3 * x1) + (2 * g1))
            break
        pass
        g1 = g1 + 1
    x2 = ((x1 // 2) - (x2)) - (5 + x2)
    x1 = (g1) * ((-g1) - (2 * x1))
    return x2


def test_program_70():
    ans = program_70(9, 6, 0)
    assert ans == 121


@hybrid
def program_71(g1):
    x1 = -1
    pass
    x1 = ((2 + g1) - (10 + g1)) * (2 + g1)
    while g1 < 1:
        pass
        x1 = (2 * x1) * ((2 + x1) + (x1 % 10))
        x1 = ((2 + x1) * (5 + g1)) - (g1)
        return x1
        g1 = g1 + 1
    x1 = ((g1 * 2) + (g1 // 2)) - (x1 * 3)
    return g1


def test_program_71():
    ans = program_71(0)
    assert ans == 3210


@hybrid
def program_72(x1, x2):
    pass
    x2 = (2 * x2) * ((x2 + 5) + (x2))
    if ((10 + x2) + (-x1)) - (x2 + 10) < 10:
        pass
        x1 = (2 * x2) - ((x2 % 2) * (2 + x1))
        return x2
    else:
        pass
        x1 = (-x1) - ((10 + x2) + (x2))
        x2 = (3 * x1) * ((5 + x1) * (5 + x1))
        return x2
    x1 = (10 + x1) + ((2 * x1) - (x2))
    x2 = ((3 * x1) * (3 * x1)) - (x2 // 10)
    return x2


def test_program_72():
    ans = program_72(0, 6)
    assert ans == 204


@hybrid
def program_73(g1):
    x1 = -1
    x2 = -1
    g2 = -1
    pass
    x1 = (2 * g1) * ((2 + g1) - (5 + g1))
    x1 = (2 + x1) - ((3 * g1) + (2 * g1))
    while g1 < 4:
        pass
        x1 = ((10 + g1) - (3 * x1)) * (3 * x1)
        x1 = ((2 + x1) + (g1 % 10)) + (g1)
        g2 = 0
        while g2 < 10:
            pass
            x1 = ((2 * g1) * (g2 + 2)) + (g1)
            x1 = ((10 + x1) * (10 + g2)) + (5 + x1)
            break
            g2 = g2 + 1
        x1 = ((10 + x1) * (10 + g1)) + (2 + g1)
        g1 = g1 + 1
    x2 = (g1 % 5) - ((3 * x1) * (3 * x1))
    return x1


def test_program_73():
    ans = program_73(0)
    assert ans == 3645


@hybrid
def program_74(x1):
    pass
    x1 = (5 + x1) + ((2 * x1) + (2 + x1))
    x1 = (2 * x1) * ((x1 + 10) + (2 + x1))
    return x1
    return x1


def test_program_74():
    ans = program_74(4)
    assert ans == 2668


@hybrid
def program_75(x1):
    pass
    x1 = (-x1) + ((5 + x1) * (5 + x1))
    x1 = ((x1 * 3) + (2 + x1)) * (2 + x1)
    return x1
    return x1


def test_program_75():
    ans = program_75(1)
    assert ans == 5254


@hybrid
def program_76(g1, g2):
    x1 = -1
    x2 = -1
    g3 = -1
    pass
    x1 = (10 + g1) * ((2 + g2) + (10 + g2))
    x2 = (10 + g1) + ((g2) + (g1))
    g3 = 0
    while g3 < 1:
        pass
        pass
        return x1
        g3 = g3 + 1
    x1 = (2 * g1) * ((3 * g1) + (-x2))
    return g1


def test_program_76():
    ans = program_76(0, 0)
    assert ans == 120


@hybrid
def program_77(x1, g1):
    pass
    x1 = ((-x1) + (g1)) - (5 + x1)
    x1 = (2 + x1) * ((3 * g1) + (x1))
    return x1


def test_program_77():
    ans = program_77(7, 0)
    assert ans == 323


@hybrid
def program_78(g1):
    x1 = -1
    pass
    x1 = (2 + g1) * ((10 + g1) - (2 + g1))
    while g1 < 7:
        pass
        x1 = (2 * g1) - ((g1) * (x1))
        x1 = ((x1 + 5) + (x1)) - (g1)
        while g1 < 9:
            pass
            x1 = (3 * g1) * ((g1 * 3) + (2 + x1))
            while g1 < 10:
                pass
                pass
                while g1 < 3:
                    pass
                    pass
                    break
                    g1 = g1 + 1
                x1 = ((x1 % 2) - (g1 * 2)) * (3 * g1)
                g1 = g1 + 1
            pass
            g1 = g1 + 1
        pass
        g1 = g1 + 1
    x1 = (x1) * ((-g1) + (10 + x1))
    return x1


def test_program_78():
    ans = program_78(0)
    assert ans == 237168


@hybrid
def program_79(x1, x2):
    x3 = -1
    pass
    x1 = (-x2) + ((10 + x2) * (x2 * 3))
    x3 = ((x2) + (10 + x2)) - (x2 % 10)
    if ((x1 % 5) + (3 * x1)) - (x2 + 5) < 10:
        pass
        pass
        return x1
    x2 = ((2 * x1) + (3 * x2)) * (x2 + 2)
    return x2


def test_program_79():
    ans = program_79(6, 9)
    assert ans == 11385


@hybrid
def program_80():
    x1 = -1
    g1 = -1
    pass
    x1 = 1
    g1 = 0
    while g1 < 2:
        pass
        x1 = (10 + g1) + ((2 * x1) * (5 + x1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    pass
    return x1


def test_program_80():
    ans = program_80()
    assert ans == 1199


@hybrid
def program_81():
    x1 = -1
    x2 = -1
    g1 = -1
    pass
    x1 = 0
    x1 = ((-x1) + (2 * x1)) + (x1 * 2)
    g1 = 0
    while g1 < 10:
        pass
        x1 = (2 * g1) + ((x1 + 2) * (5 + x1))
        while g1 < 9:
            pass
            x2 = (5 + g1) * ((2 * x1) * (3 * x1))
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        x2 = ((2 * x2) + (2 * x1)) + (3 * g1)
        g1 = g1 + 1
    x2 = (g1) * ((-g1) + (-x2))
    x1 = (3 * x1) * ((10 + g1) * (2 + g1))
    return x1


def test_program_81():
    ans = program_81()
    assert ans == 7200


@hybrid
def program_82(g1, g2):
    x1 = -1
    pass
    x1 = ((10 + g1) * (10 + g2)) - (3 * g1)
    while g1 < 3:
        pass
        pass
        if ((10 + x1) * (x1 + 2)) * (g1) < 10:
            pass
            pass
            g1 = g1 + 1
            continue
        x1 = ((g1 % 10) * (3 * x1)) + (-g1)
        g1 = g1 + 1
    x1 = (x1) + ((2 + g1) * (10 + x1))
    return x1


def test_program_82():
    ans = program_82(0, 0)
    assert ans == 10802


@hybrid
def program_83(g1, g2):
    x1 = -1
    x2 = -1
    x3 = -1
    pass
    x1 = (2 * g1) - ((5 + g1) * (2 * g2))
    if ((5 + x1) - (2 + g1)) - (5 + x1) < 10:
        pass
        x1 = ((g2 % 2) - (-g1)) - (x1)
        if (-g1) * ((3 * g2) * (2 * x1)) < 10:
            pass
            x1 = ((5 + x1) + (g1)) - (5 + g2)
            while g2 < 2:
                pass
                x1 = ((2 * x1) + (10 + x1)) * (2 + x1)
                x2 = (-x1) * ((3 * g1) + (-g2))
                g2 = g2 + 1
                continue
                g2 = g2 + 1
            pass
        pass
    else:
        pass
        x1 = (x1 // 5) - ((2 * x2) - (2 + g1))
        while g2 < 10:
            pass
            x1 = (x2 % 2) - ((x2 + 2) * (2 * g1))
            x1 = (3 * g2) * ((5 + x2) - (-g2))
            break
            g2 = g2 + 1
        x2 = ((2 * g1) * (2 + g1)) + (10 + g1)
        x2 = (3 * g2) * ((5 + g2) + (2 * g1))
    x3 = ((x1 * 2) * (2 + g2)) + (x1)
    return x1


def test_program_83():
    ans = program_83(0, 0)
    assert ans == 1540


@hybrid
def program_84(x1):
    x2 = -1
    g1 = -1
    pass
    x1 = ((-x1) + (x1 * 3)) - (-x1)
    x2 = ((x1 // 2) + (2 + x1)) + (3 * x1)
    g1 = 0
    while g1 < 10:
        pass
        x1 = (g1) * ((x1) - (x1))
        break
        g1 = g1 + 1
    x2 = ((x2 + 10) * (5 + x2)) - (10 + x1)
    x1 = ((x2 * 2) - (2 * x1)) * (g1 % 5)
    return x2


def test_program_84():
    ans = program_84(2)
    assert ans == 1316


@hybrid
def program_85(g1, g2):
    x1 = -1
    x2 = -1
    x3 = -1
    pass
    x1 = ((2 * g1) - (10 + g2)) - (g2)
    if ((g1 // 10) * (3 * x1)) - (x1) < 10:
        pass
        pass
        return g2
    else:
        pass
        x2 = (2 * g2) - ((3 * g1) - (2 + x1))
        x1 = (g2 + 5) + ((10 + x2) - (-x2))
        while g1 < 3:
            pass
            x1 = ((5 + g1) - (g1)) * (g2 // 2)
            x1 = ((g1 % 2) * (x2 + 10)) - (10 + x2)
            break
            g1 = g1 + 1
        x2 = ((2 * g2) - (10 + x2)) + (10 + g2)
        x3 = ((10 + g1) + (10 + g1)) * (5 + x2)
    pass
    return x3


def test_program_85():
    ans = program_85(0, 0)
    assert ans == 260


@hybrid
def program_86(x1, g1):
    pass
    pass
    while g1 < 6:
        pass
        pass
        break
        g1 = g1 + 1
    x1 = (10 + x1) * ((x1 // 10) + (10 + x1))
    x1 = (g1) + ((g1 // 10) + (x1 + 10))
    return x1


def test_program_86():
    ans = program_86(8, 0)
    assert ans == 334


@hybrid
def program_87(x1, g1, g2):
    g3 = -1
    pass
    x1 = (g2 * 3) - ((2 + g1) * (x1))
    x1 = ((-x1) * (2 + g2)) + (2 + g1)
    g3 = 0
    while g3 < 7:
        pass
        x1 = (g3 + 10) - ((-g1) * (x1 + 2))
        x1 = (g1 % 2) + ((10 + g1) * (x1 * 2))
        g3 = g3 + 1
        continue
        g3 = g3 + 1
    pass
    return x1


def test_program_87():
    ans = program_87(4, 0, 0)
    assert ans == 320


@hybrid
def program_88(g1):
    x1 = -1
    g2 = -1
    pass
    pass
    g2 = 0
    while g2 < 8:
        pass
        pass
        g2 = g2 + 1
        continue
        g2 = g2 + 1
    x1 = (5 + g2) - ((2 * g2) * (2 * g1))
    x1 = (x1 * 2) * ((2 + g1) + (5 + g1))
    return x1


def test_program_88():
    ans = program_88(0)
    assert ans == 182


@hybrid
def program_89(x1, x2):
    x3 = -1
    g1 = -1
    pass
    x1 = ((2 * x2) - (2 * x2)) * (x2)
    x1 = (2 * x2) - ((2 + x2) - (5 + x2))
    g1 = 0
    while g1 < 3:
        pass
        pass
        break
        g1 = g1 + 1
    x3 = (x1) * ((3 * x2) - (-x1))
    return x3


def test_program_89():
    ans = program_89(5, 3)
    assert ans == 162


@hybrid
def program_90():
    x1 = -1
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    x1 = 3
    if ((x1 * 3) - (x1 // 2)) * (x1 // 10) < 10:
        pass
        x2 = (2 * x1) * ((-x1) + (2 + x1))
        x2 = ((3 * x2) * (2 * x2)) * (10 + x1)
        g1 = 0
        while g1 < 6:
            pass
            pass
            g1 = g1 + 1
        pass
    else:
        pass
        x3 = ((2 + g1) - (-x1)) + (x1 % 2)
        return g1
    pass
    return x2


def test_program_90():
    ans = program_90()
    assert ans == 11232


@hybrid
def program_91(g1):
    x1 = -1
    x2 = -1
    pass
    x1 = (10 + g1) - ((10 + g1) - (3 * g1))
    while g1 < 10:
        pass
        x2 = ((2 * g1) - (10 + g1)) * (2 + g1)
        x1 = ((g1 % 10) + (x2)) - (-g1)
        if (g1) - ((2 + x2) * (2 * x1)) < 10:
            pass
            pass
            break
        else:
            pass
            x2 = (2 * g1) * ((2 * x1) * (x1 % 5))
            return x1
        pass
        g1 = g1 + 1
    x1 = (2 * x1) * ((10 + x1) + (x2 // 2))
    return x1


def test_program_91():
    ans = program_91(0)
    assert ans == 800


@hybrid
def program_92(x1, x2, g1, g2):
    x3 = -1
    g3 = -1
    pass
    x2 = (2 + x2) + ((10 + g1) * (x1))
    x1 = (g2 % 5) - ((g1) - (x2))
    g3 = 0
    while g3 < 6:
        pass
        x1 = (x2) + ((g2 // 5) - (2 * x1))
        g3 = g3 + 1
    x3 = (2 + g2) * ((-g1) - (10 + g1))
    x3 = ((g3) + (g1)) + (-g2)
    return x2


def test_program_92():
    ans = program_92(10, 5, 0, 0)
    assert ans == 107


@hybrid
def program_93(x1, g1):
    pass
    pass
    while g1 < 6:
        pass
        x1 = ((10 + x1) + (-x1)) + (x1 + 10)
        x1 = (x1) - ((g1 // 10) * (x1 + 10))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    x1 = ((10 + x1) + (10 + x1)) + (5 + g1)
    return x1


def test_program_93():
    ans = program_93(7, 0)
    assert ans == 285


@hybrid
def program_94():
    x1 = -1
    g1 = -1
    pass
    pass
    g1 = 0
    while g1 < 7:
        pass
        x1 = ((3 * g1) + (3 * g1)) * (3 * g1)
        x1 = (10 + x1) + ((g1) - (5 + g1))
        g1 = g1 + 1
        continue
        g1 = g1 + 1
    x1 = (5 + x1) + ((g1 + 2) * (g1 + 5))
    return x1


def test_program_94():
    ans = program_94()
    assert ans == 766


@hybrid
def program_95(x1, x2, g1):
    x3 = -1
    x4 = -1
    pass
    x1 = ((2 + g1) + (3 * x1)) + (2 * g1)
    while g1 < 3:
        pass
        x3 = ((5 + g1) * (3 * x1)) * (5 + x2)
        break
        g1 = g1 + 1
    x4 = ((5 + x3) - (2 + x2)) * (10 + x1)
    return x3


def test_program_95():
    ans = program_95(3, 4, 0)
    assert ans == 1485


@hybrid
def program_96(x1, x2, g1):
    x3 = -1
    g2 = -1
    pass
    x3 = (2 * x1) * ((-g1) - (10 + g1))
    if ((10 + x1) - (-g1)) - (5 + x1) < 10:
        pass
        x2 = ((g1) + (2 + g1)) * (x1 // 5)
        x2 = (3 * x3) - ((5 + x3) - (-g1))
        g2 = 0
        while g2 < 8:
            pass
            x2 = ((-x1) + (2 + g2)) * (x2 // 10)
            x1 = ((2 + x2) + (x1)) - (5 + x1)
            while g1 < 3:
                pass
                x3 = (10 + g2) + ((-g1) * (3 * g2))
                x3 = ((2 * x2) - (-x1)) * (2 + g2)
                return x1
                g1 = g1 + 1
            pass
            g2 = g2 + 1
        pass
    x1 = ((3 * x3) - (g1 // 2)) + (2 + x3)
    x2 = ((2 + g1) * (x1)) + (5 + x1)
    return g2


def test_program_96():
    ans = program_96(9, 9, 0)
    assert ans == 249


@hybrid
def program_97(x1):
    x2 = -1
    x3 = -1
    g1 = -1
    pass
    pass
    if (2 * x1) * ((-x1) + (x1 % 5)) < 10:
        pass
        x1 = (2 * x1) + ((x1 // 10) + (5 + x1))
        x1 = ((x1) * (x1 // 2)) * (10 + x1)
        g1 = 0
        while g1 < 10:
            pass
            x2 = ((5 + x1) + (g1 % 10)) + (10 + g1)
            g1 = g1 + 1
            continue
            g1 = g1 + 1
        pass
    else:
        pass
        x2 = (x2) + ((x1 // 2) - (g1))
        return x2
    x1 = (2 * x2) - ((g1 // 10) + (2 + x1))
    x3 = (3 * x2) * ((x2 // 2) + (3 * g1))
    return x2


def test_program_97():
    ans = program_97(2)
    assert ans == 1188


@hybrid
def program_98(g1):
    x1 = -1
    x2 = -1
    x3 = -1
    pass
    pass
    if (10 + g1) - ((-g1) + (2 + g1)) < 10:
        pass
        x1 = (2 * g1) * ((g1 + 2) - (-g1))
        while g1 < 7:
            pass
            x2 = ((x1 // 5) + (3 * x1)) * (x1 // 5)
            g1 = g1 + 1
        x1 = (5 + x2) * ((g1 * 3) * (x1 * 3))
        x1 = ((x2) - (5 + x1)) * (10 + x1)
    x2 = ((10 + x1) * (x1)) * (-x1)
    x3 = ((10 + x1) - (5 + g1)) * (x2 // 10)
    return x2


def test_program_98():
    ans = program_98(0)
    assert ans == 100000


@hybrid
def program_99(x1, g1, g2):
    x2 = -1
    x3 = -1
    g3 = -1
    pass
    x2 = (5 + g1) * ((g2 % 10) - (2 * g1))
    x1 = (5 + x1) - ((-x1) + (-g1))
    g3 = 0
    while g3 < 6:
        pass
        pass
        while g1 < 10:
            pass
            pass
            while g2 < 9:
                pass
                x2 = (10 + x1) * ((-x1) * (g3 * 2))
                x1 = (5 + g2) + ((3 * x1) * (5 + g3))
                return x1
                g2 = g2 + 1
            pass
            g1 = g1 + 1
        x1 = (2 + g2) - ((5 + g3) - (-g2))
        x3 = ((10 + x2) * (5 + g1)) - (2 * g2)
        g3 = g3 + 1
    x3 = (2 * x2) - ((g3) * (5 + x2))
    return g1


def test_program_99():
    ans = program_99(3, 0, 0)
    assert ans == 170


if __name__ == "__main__":
    test_program_0()
    test_program_1()
    test_program_2()
    test_program_3()
    test_program_4()
    test_program_5()
    test_program_6()
    test_program_7()
    test_program_8()
    test_program_9()
    test_program_10()
    test_program_11()
    test_program_12()
    test_program_13()
    test_program_14()
    test_program_15()
    test_program_16()
    test_program_17()
    test_program_18()
    test_program_19()
    test_program_20()
    test_program_21()
    test_program_22()
    test_program_23()
    test_program_24()
    test_program_25()
    test_program_26()
    test_program_27()
    test_program_28()
    test_program_29()
    test_program_30()
    test_program_31()
    test_program_32()
    test_program_33()
    test_program_34()
    test_program_35()
    test_program_36()
    test_program_37()
    test_program_38()
    test_program_39()
    test_program_40()
    test_program_41()
    test_program_42()
    test_program_43()
    test_program_44()
    test_program_45()
    test_program_46()
    test_program_47()
    test_program_48()
    test_program_49()
    test_program_50()
    test_program_51()
    test_program_52()
    test_program_53()
    test_program_54()
    test_program_55()
    test_program_56()
    test_program_57()
    test_program_58()
    test_program_59()
    test_program_60()
    test_program_61()
    test_program_62()
    test_program_63()
    test_program_64()
    test_program_65()
    test_program_66()
    test_program_67()
    test_program_68()
    test_program_69()
    test_program_70()
    test_program_71()
    test_program_72()
    test_program_73()
    test_program_74()
    test_program_75()
    test_program_76()
    test_program_77()
    test_program_78()
    test_program_79()
    test_program_80()
    test_program_81()
    test_program_82()
    test_program_83()
    test_program_84()
    test_program_85()
    test_program_86()
    test_program_87()
    test_program_88()
    test_program_89()
    test_program_90()
    test_program_91()
    test_program_92()
    test_program_93()
    test_program_94()
    test_program_95()
    test_program_96()
    test_program_97()
    test_program_98()
    test_program_99()
