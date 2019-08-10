from mnm.hybrid import hybrid


@hybrid
def level1_cfg1(x1):
    x2 = -1
    x3 = -1
    x4 = -1
    x5 = -1
    x6 = -1
    g1 = -1
    pass
    x1 = (x1) + ((-x1) + (3 * x1))
    g1 = 0
    while g1 < 4:
        pass
        x2 = ((x1 // 10) + (5 + g1)) + (2 * g1)
        while g1 < 8:
            pass
            x1 = ((g1 // 2) + (10 + x2)) * (g1 * 3)
            x3 = ((x2 + 10) + (-x2)) * (-x2)
            if (x2 + 10) + ((x2) + (5 + x2)) < 10:
                pass
                x3 = ((2 + g1) * (3 * x1)) + (g1)
            x4 = (x3) * ((2 + g1) - (-g1))
            x5 = ((2 + x1) * (3 * x3)) - (2 * x2)
            g1 = g1 + 1
        x4 = ((5 + g1) + (x1)) * (3 * x2)
        g1 = g1 + 1
    x6 = ((-x3) - (2 * x2)) - (3 * g1)
    x4 = (x2 % 2) + ((5 + x5) - (-x3))
    return g1


def test_level1_cfg1():
    ans = level1_cfg1(3)
    import ipdb
    ipdb.set_trace()
    assert ans.asnumpy() == 9


@hybrid
def level1_cfg2(x1, x2):
    pass
    x1 = (2 * x1) + ((-x1) - (10 + x2))
    x2 = (3 * x1) + ((2 * x2) - (5 + x1))
    if (-x2) * ((5 + x2) * (x1 // 10)) < 10:
        pass
        x2 = ((x1 + 10) + (3 * x1)) - (x1 % 10)
    pass
    return x2


def test_level1_cfg2():
    ans = level1_cfg2(1, 6)
    assert ans.asnumpy() == -23


@hybrid
def level1_cfg3(x1, g1, g2):
    x2 = -1
    x3 = -1
    g3 = -1
    pass
    x2 = (10 + g1) * ((5 + g2) + (g1 % 10))
    x1 = ((2 + x2) * (5 + g2)) - (5 + g1)
    g3 = 0
    while g3 < 10:
        pass
        x1 = (x2) - ((2 * g2) + (-g1))
        if ((x1 % 5) * (g3 // 10)) * (5 + g1) < 10:
            pass
            x1 = (-g2) * ((g3 % 5) + (3 * g1))
            x2 = ((x2 % 10) * (2 * x1)) + (10 + x1)
            while g2 < 1:
                pass
                x2 = (2 * g1) + ((2 * x2) * (2 + x1))
                g2 = g2 + 1
                continue
                g2 = g2 + 1
            x3 = (2 + x1) + ((-g3) + (3 * x2))
        x1 = ((g3 // 10) + (g2)) - (3 * g3)
        g3 = g3 + 1
    x3 = ((5 + x2) * (5 + x2)) - (-g1)
    x3 = ((x1) * (3 * g3)) * (-g1)
    return g1


def test_level1_cfg3():
    ans = level1_cfg3(9, 0, 0)
    assert ans.asnumpy() == 0


@hybrid
def level1_cfg4():
    x1 = -1
    g1 = -1
    pass
    x1 = 2
    g1 = 0
    while g1 < 5:
        pass
        x1 = ((-g1) - (g1 // 5)) - (x1)
        while g1 < 3:
            pass
            x1 = (g1) + ((g1 // 10) - (10 + x1))
            break
            g1 = g1 + 1
        x1 = (3 * x1) - ((10 + g1) + (3 * x1))
        g1 = g1 + 1
    x1 = (2 + x1) - ((x1 + 5) + (2 * x1))
    x1 = (-g1) - ((-g1) * (5 + g1))
    return g1


def test_level1_cfg4():
    ans = level1_cfg4()
    assert ans.asnumpy() == 5


if __name__ == "__main__":
    test_level1_cfg1()
    test_level1_cfg2()
    test_level1_cfg3()
    test_level1_cfg4()
