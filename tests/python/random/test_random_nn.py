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


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_xavier_normal(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            mnm.random.nn.xavier_normal(shape, gain=1.0)
    else:
        mnm.random.nn.xavier_normal(shape, gain=1.0)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_xavier_uniform(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            mnm.random.nn.xavier_uniform(shape, gain=1.0)
    else:
        mnm.random.nn.xavier_uniform(shape, gain=1.0)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_kaiming_normal(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            mnm.random.nn.kaiming_normal(shape)
    else:
        mnm.random.nn.kaiming_normal(shape)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_kaiming_uniform(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            mnm.random.nn.kaiming_uniform(shape)
    else:
        mnm.random.nn.kaiming_uniform(shape)


if __name__ == "__main__":
    pytest.main([__file__])
