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

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import sys
import pytest
import numpy as np

import mnm
from mnm import distributed as dist
from mnm._core.ndarray import Symbol
from mnm.testing import check, get_dist_info, skip_dist_test, run_vm_model, run_model

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max", "avg"])
def test_allreduce_with_tensor(dtype, computation):
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allreduce(x, computation=computation)
            return x

    if computation == "avg" and mnm.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = mnm.array(vx, device=device)
    run_vm_model(model, device, [vx])
    check(y, vx)
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype=dtype)
        if computation == "sum":
            target_y = ones * sum(range(1, total_rank + 1))
        elif computation == "prod":
            target_y = ones * np.prod(range(1, total_rank + 1))
        elif computation == "min":
            target_y = ones * min(1, total_rank)
        elif computation == "max":
            target_y = ones * max(1, total_rank)
        elif computation == "avg":
            target_y = ones * sum(range(1, total_rank + 1))
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_allreduce_with_tensor_list(computation):
    print("Testing allreduce with a list of tensors as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.allreduce([x1, x2], computation=computation)
            a = x[0]
            b = x[1]
            return mnm.concatenate((a, b))

    if computation == "avg" and mnm.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = model(x1, x2)
    vx1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    vx1 = mnm.array(vx1, device=device)
    vx2 = mnm.array(vx2, device=device)
    run_vm_model(model, device, [vx1, vx2])
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
        elif computation == "prod":
            sign = 1 if total_rank % 2 == 0 else -1
            target_y = np.concatenate(
                [
                    ones * np.prod(range(1, total_rank + 1)),
                    ones * sign * np.prod(range(1, total_rank + 1)),
                ]
            )
        elif computation == "min":
            target_y = np.concatenate([ones, ones * -total_rank])
        elif computation == "max":
            target_y = np.concatenate([ones * total_rank, ones * -1])
        elif computation == "avg":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)
        vy = np.concatenate([vx1.numpy(), vx2.numpy()])
        check(vy, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=4), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("rank_list", [[0], [1, 2]])
def test_allreduce_with_subcomm(dtype, rank_list):
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allreduce(x, "sum", rank_list)
            return x

    model = TestModel()
    _, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = mnm.array(vx, device=device)
    run_vm_model(model, device, [vx])
    check(y, vx)
    if rank in rank_list:
        ones = np.ones(shape=(4, 4), dtype=dtype)
        target_y = ones * sum(np.array(rank_list) + 1)
        if rank == 0:
            print(f"{rank} - Y: ", y)
            print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0, 1])
def test_allgather(axis):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allgather(x, axis=axis)
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device)
    if rank == 0:
        target_y = np.concatenate([x.numpy() * (r + 1) for r in range(total_rank)], axis=axis)
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0, 1])
def test_allgather_with_tensor_list(axis):
    print("Testing allgather with a list of tensors as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.allgather([x1, x2], axis=axis)
            return mnm.concatenate(x)

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        x1 = x1.numpy()
        x2 = x2.numpy()
        target_y1 = np.concatenate([x1 * (r + 1) for r in range(total_rank)], axis=axis)
        target_y2 = np.concatenate([x2 * (r + 1) for r in range(total_rank)], axis=axis)
        target_y = np.concatenate([target_y1, target_y2])
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=4), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("rank_list", [[0, 1], [1, 2, 3]])
def test_allgather_with_subcomm(axis, rank_list):
    print("Testing allgather with a list of tensors as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.allgather([x1, x2], axis=axis, rank_list=rank_list)
            return mnm.concatenate(x)

    model = TestModel()
    _, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    if rank in rank_list:
        x1 = np.ones(shape=(4, 4), dtype="float32")
        x2 = np.ones(shape=(4, 4), dtype="float32") * -1
        target_y1 = np.concatenate([x1 * (r + 1) for r in rank_list], axis=axis)
        target_y2 = np.concatenate([x2 * (r + 1) for r in rank_list], axis=axis)
        target_y = np.concatenate([target_y1, target_y2])
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce_scatter(computation):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = Symbol.make_tuple([x, y])
            out = mnm.reduce_scatter(z, computation=computation)
            return out

    if computation == "avg" and mnm.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    m_x, m_y = mnm.array(n_x, device=device), mnm.array(n_y, device=device)
    model.to(device=device)
    m_out = run_model(model, [m_x, m_y], device)
    if rank == 0:
        if computation == "sum":
            n_out = n_ones * sum(range(1, total_rank + 1))
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1))
        elif computation == "min":
            n_out = n_ones * min(1, total_rank)
        elif computation == "max":
            n_out = n_ones * max(1, total_rank)
        elif computation == "avg":
            n_out = n_ones * sum(range(1, total_rank + 1))
            n_out = n_out / total_rank
        else:
            assert False, "Invalid computation"
        check(m_out, n_out)
    elif rank == 1:
        if computation == "sum":
            n_out = -n_ones * sum(range(1, total_rank + 1))
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1))
        elif computation == "min":
            n_out = -n_ones * max(1, total_rank)
        elif computation == "max":
            n_out = -n_ones * min(1, total_rank)
        elif computation == "avg":
            n_out = -n_ones * sum(range(1, total_rank + 1))
            n_out = n_out / total_rank
        check(m_out, n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
def test_send_recv():
    shape = [2, 2]
    dtype = "float32"

    class TestModel_0(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            t = mnm.send(x, peer=1)
            y = mnm.recv(peer=1, shape=shape, dtype=dtype, token=t)
            out = mnm.add(x, y)
            return Symbol.make_tuple([out, t])

    class TestModel_1(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.recv(peer=0, shape=shape, dtype=dtype)
            t = mnm.send(x, peer=0, token=y)
            out = mnm.add(x, y)
            return Symbol.make_tuple([out, t])

    total_rank, rank, local_rank = get_dist_info(verbose=True)
    assert total_rank == 2, "This test only runs with 2 ranks"

    device = f"cuda({local_rank})"
    model = TestModel_0() if rank == 0 else TestModel_1()
    n_ones = np.ones(shape=shape, dtype=dtype)
    n_x = n_ones * (rank + 1)
    m_x = mnm.array(n_x, device=device)
    model.to(device=device)
    out1 = model(m_x)
    out2 = run_vm_model(model, device, [m_x])
    check(out1[0], out2[0])  # NOTE: out[1] is not set by NCCLSend currently
    n_out = n_ones * 3
    check(out1[0], n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max", "avg"])
def test_reduce(computation):
    print("Testing reduce")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.reduce(x, 0, computation=computation)
            return x

    if computation == "avg" and mnm.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")
    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = mnm.array(vx, device=device)
    vy = run_vm_model(model, device, [vx])
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = ones * sum(range(1, total_rank + 1))
        elif computation == "prod":
            target_y = ones * np.prod(range(1, total_rank + 1))
        elif computation == "min":
            target_y = ones
        elif computation == "max":
            target_y = ones * total_rank
        elif computation == "avg":
            target_y = ones * sum(range(1, total_rank + 1))
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)
        check(y, vy)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max", "avg"])
def test_reduce_list(computation):
    print("Testing reduce with list of tensor")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.reduce([x1, x2], 0, computation=computation)
            a = x[0]
            b = x[1]
            return mnm.concatenate((a, b))

    if computation == "avg" and mnm.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")
    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    vx1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    vx1 = mnm.array(vx1, device=device)
    vx2 = mnm.array(vx2, device=device)
    run_vm_model(model, device, [vx1, vx2])
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
        elif computation == "prod":
            sign = 1 if total_rank % 2 == 0 else -1
            target_y = np.concatenate(
                [
                    ones * np.prod(range(1, total_rank + 1)),
                    ones * sign * np.prod(range(1, total_rank + 1)),
                ]
            )
        elif computation == "min":
            target_y = np.concatenate([ones, ones * -total_rank])
        elif computation == "max":
            target_y = np.concatenate([ones * total_rank, ones * -1])
        elif computation == "avg":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)
        vy = np.concatenate([vx1.numpy(), vx2.numpy()])
        check(vy, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
def test_broadcast():
    print("Testing broadcast with a list of tensors.")

    # pylint: disable=attribute-defined-outside-init
    class TestModel(mnm.Model):
        def build(self, root):
            self.root = root

        @mnm.model.trace
        def forward(self, x):
            res = mnm.broadcast(x, self.root)
            return res

    model = TestModel(root=0)
    _, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = mnm.array(x, device=device)
    print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device)

    target_y = np.ones(shape=(4, 4), dtype="float32")  # rank 0's data
    print(f"{rank} - Y: ", y)
    print(f"{rank} - T: ", target_y)
    check(y, target_y)


if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
