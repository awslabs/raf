import mnm


def test_add():
    for x1 in range(-5, 5):
        for x2 in range(-5, 5):
            assert mnm.add(x1, x2) == x1 + x2
            assert mnm.add(float(x1), float(x2)) == x1 + x2


def test_subtract():
    for x1 in range(-5, 5):
        for x2 in range(-5, 5):
            assert mnm.subtract(x1, x2) == x1 - x2
            assert mnm.subtract(float(x1), float(x2)) == x1 - x2


def test_multiply():
    for x1 in range(-5, 5):
        for x2 in range(-5, 5):
            assert mnm.multiply(x1, x2) == x1 * x2
            assert mnm.multiply(float(x1), float(x2)) == x1 * x2


def test_divide():
    for x1 in range(10, 50, 5):
        for x2 in range(1, 10):
            assert mnm.divide(x1, x2) == x1 // x2
            assert mnm.divide(float(x1), float(x2)) == x1 / x2


def test_mod():
    for x1 in range(10, 50, 5):
        for x2 in range(1, 10):
            assert mnm.mod(x1, x2) == x1 % x2


def test_negative():
    for x1 in range(10):
        assert mnm.negative(x1) == -x1
        assert mnm.negative(float(x1)) == -x1


def test_less():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.less(x1, x2) == (x1 < x2)
            assert mnm.less(float(x1), float(x2)) == (x1 < x2)


def test_greater():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.greater(x1, x2) == (x1 > x2)
            assert mnm.greater(float(x1), float(x2)) == (x1 > x2)


def test_less_equal():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.less_equal(x1, x2) == (x1 <= x2)
            assert mnm.less_equal(float(x1), float(x2)) == (x1 <= x2)


def test_greater_equal():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.greater_equal(x1, x2) == (x1 >= x2)
            assert mnm.greater_equal(float(x1), float(x2)) == (x1 >= x2)


def test_equal():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.equal(x1, x2) == (x1 == x2)
            assert mnm.equal(float(x1), float(x2)) == (x1 == x2)


def test_not_equal():
    for x1 in range(3):
        for x2 in range(3):
            assert mnm.not_equal(x1, x2) == (x1 != x2)
            assert mnm.not_equal(float(x1), float(x2)) == (x1 != x2)


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
