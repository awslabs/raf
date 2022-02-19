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

# pylint: disable=no-member, no-self-use, protected-access, too-many-locals
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/op/ty/test_type_comm.py`
"""
import pytest
import numpy as np
import mnm
from mnm import distributed as dist
from mnm.testing import check_type, run_infer_type, skip_dist_test, get_dist_info
from tvm.relay import TensorType, FuncType, TupleType

SKIP_REASON = "Distribution is not enabled or #rank is not expected"


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_allreduce_with_tensor(computation):
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allreduce(x, computation=computation)
            return x

    shape = (4, 4)
    dtype = "float32"
    model = TestModel()
    _, rank, local_rank = get_dist_info()
    device = f"cuda({local_rank})"
    x = np.ones(shape=shape, dtype=dtype) * (rank + 1)
    x = mnm.array(x, device=device)
    m_func = model._internal(x).mod["main"]
    m_func = run_infer_type(m_func)
    t_a = TensorType(shape, dtype=dtype)
    t_b = TensorType(shape, dtype=dtype)
    desire_type = FuncType([t_a], t_b)
    check_type(m_func, desire_type)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_allreduce_with_tensor_list(computation):
    print("Testing allreduce with a list of tensors as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.allreduce([x1, x2], computation=computation)
            return x

    shape1 = (4, 4)
    shape2 = (3, 4, 5)
    dtype = "float32"
    model = TestModel()
    _, rank, local_rank = get_dist_info()
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=shape1, dtype=dtype) * (rank + 1)
    x2 = np.ones(shape=shape2, dtype=dtype) * (-rank - 1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    # infertype test for list of input
    m_func = model._internal(x1, x2).mod["main"]
    m_func = run_infer_type(m_func)
    t_x1 = TensorType(shape1, dtype=dtype)
    t_x2 = TensorType(shape2, dtype=dtype)
    desire_type = FuncType([t_x1, t_x2], TupleType([t_x1, t_x2]))
    check_type(m_func, desire_type)


if __name__ == "__main__":
    pytest.main([__file__])
    dist.RemoveCommunicator()
