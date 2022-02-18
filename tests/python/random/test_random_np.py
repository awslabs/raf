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
    pytest.main([__file__])
