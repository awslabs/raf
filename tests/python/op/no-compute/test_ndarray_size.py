# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

import raf

TEST_DATA = [0, 1, 90, 512, 1000, 2048, 3864, 57893, 102400]


@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_float(size):
    x = raf.array(np.random.randn(size).astype("float"))
    assert raf.ndarray_size(x) == size


@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_int(size):
    x = raf.array(np.random.randn(size).astype("int"))
    assert raf.ndarray_size(x) == size


@pytest.mark.parametrize("x", TEST_DATA[:5])
@pytest.mark.parametrize("y", TEST_DATA[:5])
def test_2d_float(x, y):
    data = raf.array(np.random.randn(x, y).astype("float"))
    assert raf.ndarray_size(data) == x * y


@pytest.mark.parametrize("x", TEST_DATA[:5])
@pytest.mark.parametrize("y", TEST_DATA[:5])
def test_2d_int(x, y):
    data = raf.array(np.random.randn(x, y).astype("int"))
    assert raf.ndarray_size(data) == x * y


@pytest.mark.parametrize("x", TEST_DATA[:4])
@pytest.mark.parametrize("y", TEST_DATA[:4])
@pytest.mark.parametrize("z", TEST_DATA[:4])
def test_3d_float(x, y, z):
    data = raf.array(np.random.randn(x, y, z).astype("float"))
    assert raf.ndarray_size(data) == x * y * z


@pytest.mark.parametrize("x", TEST_DATA[:4])
@pytest.mark.parametrize("y", TEST_DATA[:4])
@pytest.mark.parametrize("z", TEST_DATA[:4])
def test_3d_int(x, y, z):
    data = raf.array(np.random.randn(x, y, z).astype("int"))
    assert raf.ndarray_size(data) == x * y * z


if __name__ == "__main__":
    pytest.main([__file__])
