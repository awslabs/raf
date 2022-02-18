# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import numpy as np

import mnm

TEST_DATA = [0, 1, 90, 512, 1000, 2048, 3864, 57893, 102400]


@pytest.mark.parametrize("size", TEST_DATA)
def test_1d_float(size):
    x = mnm.array(np.random.randn(size).astype("float"))
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
def test_3d_int(x, y, z):
    data = mnm.array(np.random.randn(x, y, z).astype("int"))
    assert mnm.ndarray_size(data) == x * y * z


if __name__ == "__main__":
    pytest.main([__file__])
