# pylint: disable=no-self-use,invalid-name, protected-access
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import pytest
import numpy as np

import mnm
import tvm
from mnm import distributed as dist
from mnm._core.ndarray import Symbol
from mnm.testing import check, get_dist_info, skip_dist_test, run_vm_model

SKIP_REASON = "Distribution is not enabled or #rank is not expected"

def run_model(model, args, device, check_result=True):
    """Helper function to run the model using both interpreter and VM, and check if their
    results are the same. Note that some ops (e.g., reduce, send/recv) may only produce
    valid results at the target device. In this case, check_result should be skipped on
    other devices.
    """
    out1 = model(*args)
    ret = out1
    out2 = run_vm_model(model, device, args)
    if check_result:
        if not isinstance(out1, (tuple, tvm.ir.container.Array, mnm._core.value.TupleValue)):
            out1 = [out1]
            out2 = [out2]
        for o1, o2 in zip(out1, out2):
            assert check(o1, o2), "Inconsistent results between interpreter and VM at %s" % device
    return ret


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_allreduce_with_tensor(dtype, computation):
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allreduce(x, computation=computation)
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype=dtype) * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device)
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
        else:
            print("Invalid computation")
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

        @ mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.allreduce([x1, x2], computation=computation)
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
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = np.concatenate([ones * sum(range(1, total_rank + 1)),
                                       ones * -sum(range(1, total_rank + 1))])
        elif computation == "prod":
            sign = 1 if total_rank % 2 == 0 else -1
            target_y = np.concatenate([ones * np.prod(range(1, total_rank + 1)),
                                       ones * sign * np.prod(range(1, total_rank + 1))])
        elif computation == "min":
            target_y = np.concatenate([ones, ones * -total_rank])
        elif computation == "max":
            target_y = np.concatenate([ones * total_rank, ones * -1])
        else:
            print("Invalid computation")
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
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank-1)
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


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
def test_reduce_scatter():
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = Symbol.make_tuple([x, y])
            out = mnm.reduce_scatter(z)
            return out

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
        n_out = n_ones * sum(range(1, total_rank + 1))
        check(m_out, n_out)
    elif rank == 1:
        n_out = -n_ones * sum(range(1, total_rank + 1))
        check(m_out, n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True),
                    reason=SKIP_REASON)
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
    n_x = n_ones * (rank+1)
    m_x = mnm.array(n_x, device=device)
    model.to(device=device)
    m_out = run_model(model, [m_x], device, check_result=bool(rank > 0))
    m_out = m_out[0]
    if rank > 0:
        n_out = n_ones * 3
        check(m_out, n_out)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce(computation):
    print("Testing reduce")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.reduce(x, 0, computation=computation)
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device, check_result=bool(rank == 0))
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
        else:
            print("Invalid computation")
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce_list(computation):
    print("Testing reduce with list of tensor")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):
            x = mnm.reduce([x1, x2], 0, computation=computation)
            return mnm.concatenate(x)

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank-1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = np.concatenate([ones * sum(range(1, total_rank + 1)),
                                       ones * -sum(range(1, total_rank + 1))])
        elif computation == "prod":
            sign = 1 if total_rank % 2 == 0 else -1
            target_y = np.concatenate([ones * np.prod(range(1, total_rank + 1)),
                                       ones * sign * np.prod(range(1, total_rank + 1))])
        elif computation == "min":
            target_y = np.concatenate([ones, ones * -total_rank])
        elif computation == "max":
            target_y = np.concatenate([ones * total_rank, ones * -1])
        else:
            print("Invalid computation")
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


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
    x = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x = mnm.array(x, device=device)
    print(f"{rank} - X: ", x)
    model.to(device=device)
    y = run_model(model, [x], device)

    target_y = np.ones(shape=(4, 4), dtype="float32")  # rank 0's data
    print(f"{rank} - Y: ", y)
    print(f"{rank} - T: ", target_y)
    check(y, target_y)


if __name__ == "__main__":
    pytest.main([__file__])
    dist.RemoveCommunicator()
