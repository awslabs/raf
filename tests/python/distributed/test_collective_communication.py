# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches, redefined-outer-name
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import os
import sys
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf._core.ndarray import Symbol
from raf.testing import check, get_dist_comm_info, skip_dist_test, run_vm_model, run_model

SKIP_REASON = "Distribution is not enabled or #rank is not expected"


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max", "avg"])
def test_allreduce_with_tensor(dtype, computation):
    """Testing allreduce with a single tensor as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.allreduce(x, computation=computation)
            return x

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = raf.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = raf.array(vx, device=device)
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
    """Testing allreduce with a list of tensors as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.allreduce([x1, x2], computation=computation)
            a = x[0]
            b = x[1]
            return raf.concatenate((a, b))

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = model(x1, x2)
    vx1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    vx1 = raf.array(vx1, device=device)
    vx2 = raf.array(vx2, device=device)
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
@pytest.mark.parametrize("rank_list", [[[0, 1], [2, 3]], [[1, 2, 3]]])
def test_allreduce_with_subcomm(dtype, rank_list):
    """Testing allreduce with a single tensor as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.allreduce(x, "sum", rank_list)
            return x

    model = TestModel()
    _, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = raf.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = raf.array(vx, device=device)
    run_model(model, [vx], device)
    check(y, vx)
    for group in rank_list:
        if rank in group:
            ones = np.ones(shape=(4, 4), dtype=dtype)
            target_y = ones * sum(np.array(group) + 1)
            if rank == 0:
                print(f"{rank} - Y: ", y)
                print(f"{rank} - T: ", target_y)
            check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0, 1])
def test_allgather(axis):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.allgather(x, axis=axis)
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = raf.array(x, device=device)
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
    """Testing allgather with a list of tensors as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.allgather([x1, x2], axis=axis)
            return raf.concatenate(x)

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
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
@pytest.mark.parametrize("rank_list", [[[0, 1], [2, 3]], [[1, 2, 3]]])
def test_allgather_with_subcomm(axis, rank_list):
    """Testing allgather with a list of tensors as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.allgather([x1, x2], axis=axis, rank_list=rank_list)
            return raf.concatenate(x)

    model = TestModel()
    _, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    for group in rank_list:
        if rank in group:
            x1 = np.ones(shape=(4, 4), dtype="float32")
            x2 = np.ones(shape=(4, 4), dtype="float32") * -1
            target_y1 = np.concatenate([x1 * (r + 1) for r in group], axis=axis)
            target_y2 = np.concatenate([x2 * (r + 1) for r in group], axis=axis)
            target_y = np.concatenate([target_y1, target_y2])
            print(f"{rank} - Y: ", y)
            print(f"{rank} - T: ", target_y)
            check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce_scatter(computation):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = Symbol.make_tuple([x, y])
            out = raf.reduce_scatter(z, computation=computation)
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    m_x, m_y = raf.array(n_x, device=device), raf.array(n_y, device=device)
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


@pytest.mark.skipif(skip_dist_test(min_rank_num=4, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
@pytest.mark.parametrize("rank_list", [[[0, 1], [2, 3]]])
def test_reduce_scatter_with_rank_list(computation, rank_list):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            z = Symbol.make_tuple([x, y])
            out = raf.reduce_scatter(z, computation=computation, rank_list=rank_list)
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    m_x, m_y = raf.array(n_x, device=device), raf.array(n_y, device=device)
    model.to(device=device)
    m_out = run_model(model, [m_x, m_y], device)
    for group in rank_list:
        if rank in group:
            ones = np.ones(shape=(4, 4), dtype="float32")
            if computation == "sum":
                even_out = ones * sum(np.array(group) + 1)
                odd_out = -even_out
            elif computation == "prod":
                even_out = ones * np.prod([(temp_rank + 1) for temp_rank in np.array(group)])
                odd_out = even_out
            elif computation == "min":
                even_out = ones * min(np.array(group) + 1)
                odd_out = -ones * max(np.array(group) + 1)
            elif computation == "max":
                even_out = ones * max(np.array(group) + 1)
                odd_out = -ones * min(np.array(group) + 1)
            elif computation == "avg":
                even_out = ones * sum(np.array(group) + 1)
                even_out = even_out / total_rank
                odd_out = -even_out
            if rank % 2 == 0:
                check(m_out, even_out)
            else:
                check(m_out, odd_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
def test_send_recv():
    shape = [2, 2]
    dtype = "float32"

    class TestModel_0(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            t = raf.send(x, peer=1)
            y = raf.recv(peer=1, shape=shape, dtype=dtype, token=t)
            out = raf.add(x, y)
            return Symbol.make_tuple([out, t])

    class TestModel_1(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.recv(peer=0, shape=shape, dtype=dtype)
            t = raf.send(x, peer=0, token=y)
            out = raf.add(x, y)
            return Symbol.make_tuple([out, t])

    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    assert total_rank == 2, "This test only runs with 2 ranks"

    device = f"cuda({local_rank})"
    model = TestModel_0() if rank == 0 else TestModel_1()
    n_ones = np.ones(shape=shape, dtype=dtype)
    n_x = n_ones * (rank + 1)
    m_x = raf.array(n_x, device=device)
    model.to(device=device)
    out1 = model(m_x)
    out2 = run_vm_model(model, device, [m_x])
    check(out1[0], out2[0])  # NOTE: out[1] is not set by NCCLSend currently
    n_out = n_ones * 3
    check(out1[0], n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max", "avg"])
def test_reduce(computation):
    """Testing reduce"""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf.reduce(x, 0, computation=computation)
            return x

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")
    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = raf.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx = raf.array(vx, device=device)
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
    """Testing reduce with list of tensor"""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.reduce([x1, x2], 0, computation=computation)
            a = x[0]
            b = x[1]
            return raf.concatenate((a, b))

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")
    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    vx1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    vx1 = raf.array(vx1, device=device)
    vx2 = raf.array(vx2, device=device)
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
    """Testing broadcast with a list of tensors."""

    # pylint: disable=attribute-defined-outside-init
    class TestModel(raf.Model):
        def build(self, root):
            self.root = root

        @raf.model.trace
        def forward(self, x):
            res = raf.broadcast(x, self.root)
            return res

    model = TestModel(root=0)
    _, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = raf.array(x, device=device)
    print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device)

    target_y = np.ones(shape=(4, 4), dtype="float32")  # rank 0's data
    print(f"{rank} - Y: ", y)
    print(f"{rank} - T: ", target_y)
    check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0])
def test_group_allgather(axis):
    """Testing allgather with a list of tensors as input."""

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2, y1, y2):
            out = raf.group_allgather([x1, x2], axis, [y1, y2])
            return out[0], out[1]

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    y1 = np.ones(shape=(8, 4), dtype="float32")
    y2 = np.ones(shape=(8, 4), dtype="float32")
    y1 = raf.array(y1, device=device)
    y2 = raf.array(y2, device=device)

    model.to(device=device)
    run_model(model, [x1, x2, y1, y2], device)

    if rank == 0:
        x1 = x1.numpy()
        x2 = x2.numpy()
        target_y1 = np.concatenate([x1 * (r + 1) for r in range(total_rank)], axis=axis)
        target_y2 = np.concatenate([x2 * (r + 1) for r in range(total_rank)], axis=axis)
        check(y1, target_y1)
        check(y2, target_y2)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_group_reduce_scatter(computation):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            out = raf.group_reduce_scatter([x, y], computation)
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    m_x, m_y = raf.array(n_x, device=device), raf.array(n_y, device=device)
    model.to(device=device)
    m_out = run_model(model, [m_x, m_y], device)
    if rank == 0:
        n_ones = np.ones(shape=(2, 4), dtype="float32")
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
        check(m_out[0], n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce_scatter_single_tensor(computation):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            z = Symbol.make_tuple([x])
            out = raf.reduce_scatter(z, computation=computation)
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_comm_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * rank
    m_x = raf.array(n_x, device=device)
    model.to(device=device)
    m_out = run_model(model, [m_x], device)
    out_shape = (2, 4)
    if computation == "sum":
        n_out = np.ones(out_shape, dtype="float32")
    elif computation == "prod":
        n_out = np.zeros(out_shape, dtype="float32")
    elif computation == "min":
        n_out = np.zeros(out_shape, dtype="float32")
    elif computation == "max":
        n_out = np.ones(out_shape, dtype="float32")
    elif computation == "avg":
        n_out = np.ones(out_shape, dtype="float32")
        n_out = n_out / total_rank
    else:
        assert False, "Invalid computation"
    check(m_out, n_out)


if __name__ == "__main__":
    if os.environ.get("RAF_FILE_STORE_PATH", None):
        dist.set_default_communicator("void")
        comm = dist.get_communicator()
        size = int(
            os.environ.get("OMPI_COMM_WORLD_SIZE", None) or os.environ.get("MPIRUN_NPROCS", None)
        )
        rank = int(
            os.environ.get("OMPI_COMM_WORLD_RANK", None) or os.environ.get("MPIRUN_RANK", None)
        )
        comm.size = size
        comm.rank = rank
        comm.local_size = size
        comm.local_rank = rank
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
