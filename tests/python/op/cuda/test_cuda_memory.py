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

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
import pytest
import numpy as np

import mnm
from mnm.testing import check, with_dialect
from mnm.testing.utils import run_model


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [(64, 128), (128, 256)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tensor_fusion_and_defusion(shape, dtype):
    size = 1
    for axis in shape:
        size *= axis
    sizes = [size, size]
    tuple_shape = shape + shape
    shape_indices = [len(shape), 2 * len(shape)]

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x_1 = [x, x]
            x_2 = mnm.fuse_tensor(x_1)
            x_3 = mnm.defuse_tensor(x_2, sizes, tuple_shape, shape_indices)
            return x_3[0]

    x = np.ones(shape, dtype=dtype)
    x = mnm.array(x, device="cuda(0)")
    model = Model()
    device = f"cuda(0)"
    y_c = run_model(model, [x], device)
    rtol = 1e-4 if dtype == "float32" else 4e-2
    atol = 1e-4 if dtype == "float32" else 4e-2
    check(x, y_c, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
