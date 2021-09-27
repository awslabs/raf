# Distributed Training With Meta

This tutorial introduces how to train your model with multiple GPUs.
To enable distributed training, you need to turn `MNM_USE_CUDA`, `MNM_USE_MPI`, and `MNM_USE_NCCL` on in `${MNM_HOME}/build/config.cmake` before cmake.

We will first introduce collective communication operators for distribution, followed by data parallel and ZeRO optimizations.

## Resources for Collective Communication Operators

Using collective communication operators like `allreduce`, you must take the network resources.
In the current design, theses operators need `Connector` and `Communicator`.

### Connector

The `Connector` is used to establish connection between workers. It provides information including size (the number of workers in the cluster), rank, local_size(the number of workers in this node), local_rank. Meanwhile, it provides some methods like `Barrier()` and `Broadcast(*)`.

Currenly, we have implemented `MPIConnector`, which will use mpi to establish connection. (You must install MPI in your machine to enable this Connector).

### Communicator

The `Communicator` is used to transfer data and information between workers. Each collective_comm operator will request a communicator before execution. The communicator will tell the operator how to and where to tranfer the data.

## Collective Communication Operators

You can check the current supported collective communication operators by querying the
distributed op:

```python
pyhton3
>>> import mnm
>>> dir(mnm.distributed.op)
```

Here we use `AllReduce` as an example to show how to use collective communication operators `AllReduce` takes an `ndarray` or list of `ndarray` as input, and return the aggregated data.
You can check the user guides of NVIDIA Collective Communications Library (NCCL) for the detail
functions of each operators: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html.

Before you use the operator, you must get a distributed context, from which you will know which GPU is owned by the current process.

``` python
import numpy as np
import mnm
from mnm import distributed as dist

dctx = dist.get_context()
root_rank = dctx.root_rank     # The root rank.
size = dctx.size               # The total available ranks.
rank = dctx.rank               # The rank of this process, ranging from 0 to (size - 1).
local_rank = dctx.local_rank   # The local rank of this process on this instance.
local_size = dctx.local_size   # The local rank number of this instance.

device = f'cuda({local_rank})' # Let this process control the rank-th GPU on this instance.
```

Then you can use allreduce operators.

``` python
x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
x = mnm.array(x, device=device)
print("Before allreduce: ", x)
x = mnm.allreduce(x) # or: x = mnm.allreduce([x])
print("After allreduce : ", x)
```

## Distributed Training

To enable distributed training, you need to set the corresponding flags in the distributed context. For example:

```python
import mnm
from mnm import distributed as dist
dctx = dist.get_context()
dctx.enable_data_parallel = True
```

Note that if you are using the provided script (i.e., `dist_example.py`), you can simply change the values in `meta_dist_config`. We will introduce each configure in the following subsections along with the distribution methodologies.

Since now we rely on MPI to manage multi-processing, we need to launch the script
with `mpirun`:

```bash
# Run training on a single machine with 4 GPUs.
mpirun -np 4 python3 scripts/distributed/dist_example.py

# Run training on 2 machines with 4 GPUs each.
mpirun -H node1:4,node2:4 python3 scripts/distributed/dist_example.py

# Using hostfile to specify hosts and number of GPUs on each host.
mpirun -np 8 --hostfile my_hosts.txt python3 scripts/distributed/dist_example.py
```

If you are using OpenMPI, the hostfile my_hosts.txt will be like:

```txt
node1 slots=4
node2 slots=4
```

If you are using MPICH, the hostfile my_hosts.txt will be like:

```txt
node1:4
node2:4
```

### Data Parallelism

Data parallelism distributes the input training data to each device, and performs
`AllReduce` on gradients.

To enable data parallelism, set the corresponding configure to be `True` in the script:

```python
meta_dist_config = {
    "enable_data_parallel": True
}
```

### ZeRO Optimizations

ZeRO optimizations are introduced in this paper https://arxiv.org/abs/1910.02054. It has 3 stages:
- ZeRO-1: Partition the optimizer status, such as variants and momentum, so that each device only needs to own a partition of optimizer status, reducing the memory footprint of optimizer status by 1/N, where N is the total number of working devices.
- ZeRO-2: Based on ZeRO-1, but replaces `AllReduce` by `ReduceAndScatter` to further reduce the gradient memory footprint.
- ZeRO-3: Based on ZeRO-2, further partition the learnable weights (not supported yet).

To enable ZeRO, again we just need to set the corresponding configure in the script:

```python
meta_dist_config = {
    "zero_opt_level": 1, # Use ZeRO-1
    ...
}
```

### Possible Improvements

* Fuse small tensors to reduce communication overhead.
