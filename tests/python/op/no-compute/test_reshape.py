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


@pytest.mark.parametrize("shapes", [((4, 4), (4, 2)), ((5, 3), (5, 5))])
def test_reshape_error(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    with pytest.raises(ValueError):
        mnm.reshape(x, reshape)


@pytest.mark.parametrize(
    "shapes", [((4, 4, 4), (4, 2, 8)), ((5, 3, 2), (5, 6)), ((5, 6), (3, 2, 5))]
)
def test_reshape(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    y = mnm.reshape(x, reshape)
    assert y.shape == reshape


def test_create_view_with_empty_shape():
    x = np.random.randn(1).astype("float32")
    x = mnm.array(x)
    y = mnm.squeeze(x)
    assert y.shape == ()


if __name__ == "__main__":
    pytest.main([__file__])
