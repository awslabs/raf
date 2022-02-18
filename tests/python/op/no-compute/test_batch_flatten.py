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

import functools
import operator
import pytest
import numpy as np

import mnm
from mnm.testing import run_vm_model, check, get_testable_devices


@pytest.mark.parametrize("shape", [(1, ()), (5, (5,))])
def test_batch_flatten_error(shape):
    shape, reshape = shape
    x = np.random.randn(shape).astype("float32").reshape(reshape)
    x = mnm.array(x)
    assert x.shape == reshape
    with pytest.raises(ValueError):
        mnm.batch_flatten(x)


@pytest.mark.parametrize("shape", [[5, 3], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_batch_flatten(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.batch_flatten(x)

    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    # imperative
    y_i = mnm.batch_flatten(x)
    expected = (5, functools.reduce(operator.mul, list(x.shape)[1:]))
    assert y_i.shape == expected
    dy = mnm.reshape(y_i, mnm.shape(x))
    assert dy.shape == x.shape
    assert (x.numpy() == dy.numpy()).all()
    # traced
    model = Model()
    y_t = run_vm_model(model, device, [x])
    check(y_t, y_i)


if __name__ == "__main__":
    pytest.main([__file__])
