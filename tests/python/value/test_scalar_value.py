from mnm._ffi.value import FloatValue, IntValue


def test_int_value():
    value = IntValue(1)
    data = value.data
    assert data == 1


def test_float_value():
    value = FloatValue(3.1415926535897932384626)
    data = value.data
    assert data == 3.1415926535897932384626


if __name__ == "__main__":
    test_int_value()
    test_float_value()
