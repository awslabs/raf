import pytest

import mnm


def _shape(shape):
    if shape is None:
        return ()
    return tuple(shape)


@pytest.mark.parametrize("shape", [None, [], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_uniform(shape):
    assert mnm.random.uniform(shape=shape).shape == _shape(shape)


@pytest.mark.parametrize("shape", [None, [], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_normal(shape):
    assert mnm.random.normal(shape=shape).shape == _shape(shape)


if __name__ == "__main__":
    pytest.main()
