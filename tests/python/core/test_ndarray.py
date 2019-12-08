import mnm


def test_requires_grad():
    a = mnm.array([1, 2, 3], dtype="float32")
    assert not a.requires_grad
    bools = [False, True, True, False, True, False, False, False, True]
    for val in bools:
        a.requires_grad = val
        assert a.requires_grad == val


if __name__ == "__main__":
    test_requires_grad()
