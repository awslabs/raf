import pytest
import numpy as np

import mnm

TEST_DATA = [0, 1, 90, 512, 1000, 2048, 3864, 57893, 102400]

@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_float(size):
    x = mnm.array(np.random.randn(size).astype("float"))
    assert mnm.ndarray_size(x) == size

@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_double(size):
    x = mnm.array(np.random.randn(size).astype("double"))
    assert mnm.ndarray_size(x) == size

@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_int(size):
    x = mnm.array(np.random.randn(size).astype("int"))
    assert mnm.ndarray_size(x) == size

@pytest.mark.parametrize("x", TEST_DATA[:5])
@pytest.mark.parametrize("y", TEST_DATA[:5])
def test_2d_float(x, y):
    data = mnm.array(np.random.randn(x, y).astype("float"))
    assert mnm.ndarray_size(data) == x * y

@pytest.mark.parametrize("x", TEST_DATA[:5])
@pytest.mark.parametrize("y", TEST_DATA[:5])
def test_2d_double(x, y):
    data = mnm.array(np.random.randn(x, y).astype("double"))
    assert mnm.ndarray_size(data) == x * y

@pytest.mark.parametrize("x", TEST_DATA[:5])
@pytest.mark.parametrize("y", TEST_DATA[:5])
def test_2d_int(x, y):
    data = mnm.array(np.random.randn(x, y).astype("int"))
    assert mnm.ndarray_size(data) == x * y

@pytest.mark.parametrize("x", TEST_DATA[:4])
@pytest.mark.parametrize("y", TEST_DATA[:4])
@pytest.mark.parametrize("z", TEST_DATA[:4])
def test_3d_float(x, y, z):
    data = mnm.array(np.random.randn(x, y, z).astype("float"))
    assert mnm.ndarray_size(data) == x * y * z

@pytest.mark.parametrize("x", TEST_DATA[:4])
@pytest.mark.parametrize("y", TEST_DATA[:4])
@pytest.mark.parametrize("z", TEST_DATA[:4])
def test_3d_double(x, y, z):
    data = mnm.array(np.random.randn(x, y, z).astype("double"))
    assert mnm.ndarray_size(data) == x * y * z

@pytest.mark.parametrize("x", TEST_DATA[:4])
@pytest.mark.parametrize("y", TEST_DATA[:4])
@pytest.mark.parametrize("z", TEST_DATA[:4])
def test_3d_int(x, y, z):
    data = mnm.array(np.random.randn(x, y, z).astype("int"))
    assert mnm.ndarray_size(data) == x * y * z

if __name__ == "__main__":
    pytest.main([__file__])
