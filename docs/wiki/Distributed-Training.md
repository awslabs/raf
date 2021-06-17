# Distributed Training With Meta

This blog will introduce how to train your model with Data Parallel.
To enable distributed training, you need to turn the `MNM_USE_MPI` and `MNM_USE_NCCL` on in `${MNM_HOME}/build/config.cmake` before cmake.
Next, we will introduce related operators and AutoParallel(still in progress) separately.

## Resources for communication Operators

Using collective communication operators like allreduce, you must take the network resources.
In the current design, theses operators need `Connector` and `Communicator`.

### Connector

The `Connector` is used to establish connection between workers. It provides information including size (the number of workers in the cluster), rank, local_size(the number of workers in this node), local_rank. Meanwhile, it provides some methods like `Barrier()` and `Broadcast(*)`.

Currenly, we have implemented `MPIConnector`, which will use mpi to establish connection. (You must install mpi in your machine to enable this Connector).

### Communicator

The `Communicator` is used to transfer data and information between workers. Each collective_comm operator will request a communicator before execution. The communicator will tell the operator how to and where to tranfer the data.

## Operators

### AllReduce

AllReduce is a common collective communication operator. It takes an `ndarray` or list of `ndarray` as input, and return the aggregated data.

### Examples

Before you use the operator, you must get a distributed context, from which you will know which GPU is owned by the current process.

``` python
import numpy as np
import mnm
from mnm import distributed as dist

dctx = dist.get_context()
root_rank = dctx.root_rank
rank = dctx.rank
size = dctx.size
local_rank = dctx.local_rank
local_size = dctx.local_size

device = f'cuda({local_rank})'
```

Then you can use allreduce operators.

``` python
x = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
x = mnm.array(x, device=device)
print("Before allreduce: ", x)
x = mnm.allreduce(x) # or: x = mnm.allreduce([x])
print("After allreduce : ", x)
```

## Auto Parallel - Data Parallel Part

### How to use

To use auto data parallel, you just need to set `mnm.distributed.get_context().enable_data_parallel = True` before running model.

What need to be mentioned is that currently meta run operators synchronously, as we will call `stream->wait()` after each op. If you want to overlap the communication and computation, you need to make some change to src code: add a if-else statement before comment the code `req->stream[i].stream->Wait();` in `interpreter.cc::Interpreter::InvokePrimitiveOpEnv()`.  

``` cpp
if (op->name != "mnm.op.comm.allreduce") req->stream[i].stream->Wait();
```

*note: We have ensured that communication operators will run on specific stream (different with computation operators).*

To launch a data parallel training job, currenlt you can use mpirun, for exmaple:

```bash
# The following command will run data parallel training on single machine with 4 gpus.
mpirun -np 4 python3 scripts/distributed/dist_example.py

# The following command will run data parallel training on 2 machines with 4 gpus each.
mpirun -H node1:4,node2:4 python3 scripts/distributed/dist_example.py
# or using hostfile to specify hosts and number of gpus on each hosts.
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

### Basic design

Design a new Pass for the IR. Add a communication operator (eg. allreduce) after the gradient of a parameter is generated.

What's more, we will add a new operator called `stream_sync` before return of backward to make sure that all the communication tasks have finished.

### Advanced design

See [PR:dev-dist](https://github.com/meta-project/meta/pull/201) for details if you want.

### TODO

These todos will be implemented in following PRs.

* Auto Model Parallel
* Overlapping communication with forward propagation (of next iteration)
* Fuse small tensors when communicating. (Could Pass::FuseOps support this?)
