"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import pytest
import numpy as np

import mnm
from mnm.distributed import DistContext


def check(m_x, n_x, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(m_x.asnumpy(), n_x, rtol=rtol, atol=atol)


def get_node_info():
    dctx = DistContext.get_context()
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
def test_allreduce_with_tensor():
    print("Testing allreduce with a single tensor as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use,invalid-name
            x = mnm.allreduce(x)
            return x

    model = TestModel()
    rank, local_rank = get_node_info()
    ctx = f"cuda({local_rank})"
    x = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x = mnm.array(x, ctx=ctx)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(ctx=ctx)
    y = model(x)
    if rank == 0:
        target_y = x.asnumpy() * (1+2)
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skip()
def test_allreduce_with_tensor_list():
    print("Testing allreduce with a list of tensors as input.")

    class TestModel(mnm.Model):
        def build(self):
            pass

        @ mnm.model.trace
        def forward(self, x1, x2):  # pylint: disable=no-self-use
            x = mnm.allreduce([x1, x2])  # pylint: disable=invalid-name
            return mnm.concatenate(x)

    model = TestModel()
    rank, local_rank = get_node_info()
    ctx = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank+1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank-1)
    x1 = mnm.array(x1, ctx=ctx)
    x2 = mnm.array(x2, ctx=ctx)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(ctx=ctx)
    y = model(x1, x2)
    if rank == 0:
        target_y = mnm.concatenate([x1, x2]).asnumpy() * (1+2)
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


if __name__ == "__main__":
    if mnm.build.with_distributed():
        test_allreduce_with_tensor()
        test_allreduce_with_tensor_list()
