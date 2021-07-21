# pylint: disable=no-self-use,invalid-name
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import pytest
import numpy as np

import mnm
from mnm import distributed as dist
from mnm._core.ndarray import Symbol


def check(m_x, n_x, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(m_x.numpy(), n_x, rtol=rtol, atol=atol)


def get_node_info():
    dctx = dist.get_context()
    root_rank = dctx.root_rank
    rank = dctx.rank
    size = dctx.size
    local_rank = dctx.local_rank
    local_size = dctx.local_size

    if rank == 0:
        node_info = f"root_rank={root_rank},rank={rank}, \
        size={size},local_rank={local_rank}, local_size={local_size} "
        print(node_info)
    return rank, local_rank


@pytest.mark.skip()
def test_allreduce_with_tensor(computation):
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allreduce(x, computation=computation)
            return x

    model = TestModel()
    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    if rank == 0:
        target_y = 0
        if computation == "sum":
            target_y = x.numpy() * (1 + 2)
        elif computation == "prod":
            target_y = x.numpy() * (1 * 2)
        elif computation == "min":
            target_y = x.numpy() * min(1, 2)
        elif computation == "max":
            target_y = x.numpy() * max(1, 2)
        else:
            print("Invalid computation")
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skip()
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
    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank-1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = model(x1, x2)
    if rank == 0:
        target_y = 0
        x1 = x1.numpy()
        x2 = x2.numpy()
        if computation == "sum":
            target_y = np.concatenate([x1 * (1+2), x2  * (-1) * ((-1)+(-2))])
        elif computation == "prod":
            target_y = np.concatenate([x1 * (1*2), x2  * (-1) * ((-1)*(-2))])
        elif computation == "min":
            target_y = np.concatenate([x1 * min(1, 2), x2  * (-1) * min(-1, -2)])
        elif computation == "max":
            target_y = np.concatenate([x1 * max(1, 2), x2  * (-1) * max(-1, -2)])
        else:
            print("Invalid computation")
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skip()
def test_allgather(axis):
    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            x = mnm.allgather(x, axis=axis)
            return x

    model = TestModel()
    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x = mnm.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    if rank == 0:
        target_y = np.concatenate([x.numpy(), x.numpy() * 2], axis=axis)
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skip()
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
    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank-1)
    x1 = mnm.array(x1, device=device)
    x2 = mnm.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = model(x1, x2)
    if rank == 0:
        x1 = x1.numpy()
        x2 = x2.numpy()
        target_y1 = np.concatenate([x1, x1 * 2], axis=axis)
        target_y2 = np.concatenate([x2, x2 * 2], axis=axis)
        target_y = np.concatenate([target_y1, target_y2])
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skip()
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
    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=(4, 4), dtype="float32")
    n_x = n_ones * (rank+1)
    n_y = -n_ones * (rank+1)
    m_x, m_y = mnm.array(n_x, device=device), mnm.array(n_y, device=device)
    model.to(device=device)
    m_out = model(m_x, m_y)
    if rank == 0:
        n_out = n_ones * 3
        check(m_out, n_out)
    elif rank == 1:
        n_out = -n_ones * 3
        check(m_out, n_out)

@pytest.mark.skip()
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

    rank, local_rank = get_node_info()
    device = f"cuda({local_rank})"
    model = TestModel_0() if rank == 0 else TestModel_1()
    n_ones = np.ones(shape=shape, dtype=dtype)
    n_x = n_ones * (rank+1)
    m_x = mnm.array(n_x, device=device)
    model.to(device=device)
    m_out = model(m_x)
    m_out = m_out[0]
    if rank == 1:
        n_out = n_ones * 3
        check(m_out, n_out)


if __name__ == "__main__":
    if mnm.build.with_distributed():
        test_reduce_scatter()
        test_allreduce_with_tensor(computation="sum")
        test_allreduce_with_tensor(computation="prod")
        test_allreduce_with_tensor(computation="min")
        test_allreduce_with_tensor(computation="max")
        test_allreduce_with_tensor_list(computation="sum")
        test_allreduce_with_tensor_list(computation="prod")
        test_allreduce_with_tensor_list(computation="min")
        test_allreduce_with_tensor_list(computation="max")
        test_allgather(axis=0)
        test_allgather(axis=1)
        test_allgather_with_tensor_list(axis=0)
        test_allgather_with_tensor_list(axis=1)
        test_send_recv()
        dist.RemoveCommunicator()
