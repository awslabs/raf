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

# pylint: disable=no-self-use, protected-access, too-many-locals
import pytest
import numpy as np
import mnm
from mnm.testing import check_type, run_infer_type, get_dist_info, with_dialect
from tvm.relay import TensorType, FuncType, TupleType


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [(128, 128), (64, 64), (32, 64, 128)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_fuse_tensor_type(shape, dtype, n):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.fuse_tensor([x] * n)
            return y

    model = Model()
    _, _, local_rank = get_dist_info()
    device = f"cuda({local_rank})"
    x = np.ones(shape=shape, dtype=dtype)
    x = mnm.array(x, device=device)
    m_func = model._internal(x).mod["main"]
    m_func = run_infer_type(m_func)
    t_a = TensorType(shape, dtype=dtype)
    size = 1
    for axis in shape:
        size *= axis
    size *= n
    t_b = TensorType((size,), dtype=dtype)
    desire_type = FuncType([t_a], t_b)
    check_type(m_func, desire_type)


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "shapes,shape_indices",
    [
        [(32, 32, 64, 64), (2, 4)],
        [(16, 32, 64, 64, 128), (2, 5)],
        [(32, 32, 64, 64, 32), (3, 5)],
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_defuse_tensor_type(shapes, shape_indices, dtype):
    tuple_shape = []
    tensor_types = []
    total_size = 0
    sizes = []
    start_index = 0
    for end_index in shape_indices:
        size = 1
        for axis in shapes[start_index:end_index]:
            size *= axis
        total_size += size
        sizes.append(size)
        tuple_shape.append(tuple(shapes[start_index:end_index]))
        tensor_types.append(TensorType(shapes[start_index:end_index], dtype=dtype))
        start_index = end_index

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.defuse_tensor(x, sizes, shapes, shape_indices)
            return y

    model = Model()
    _, _, local_rank = get_dist_info()
    device = f"cuda({local_rank})"
    x = np.ones(shape=(total_size,), dtype=dtype)
    x = mnm.array(x, device=device)
    m_func = model._internal(x).mod["main"]
    m_func = run_infer_type(m_func)
    t_a = TensorType((total_size,), dtype=dtype)
    t_b = TupleType(tensor_types)
    desire_type = FuncType([t_a], t_b)
    check_type(m_func, desire_type)


if __name__ == "__main__":
    pytest.main([__file__])
