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

import numpy as np
import pytest

import mnm


def test_mnm_array_cpu():
    array = mnm.array([1, 2, 3], dtype="int8", device="cpu")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_mnm_array_cuda():
    array = mnm.array([1, 2, 3], dtype="int8", device="cuda")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


if __name__ == "__main__":
    test_mnm_array_cpu()
    test_mnm_array_cuda()
