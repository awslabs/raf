<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Distribution Mechanism

This article explains the distribution mechanism in RAF.
For the tutorial of distributed training, please see [here](../2_user_guide/Distributed-Training.md)

## Overview

Distributed RAF mainly adopts the Single Program Multiple Data (SPMD) programming model driven by MPI-like libraries such as NVIDIA Collective Communications Library (NCCL), which provides a set of collective communication operators and bridges multiple discrete GPUs on numerous servers connected via conventional Ethernet or high-performance Infiniband. Specifically, RAF utilizes both MPI and NCCL in a mixed way, where MPI is used to launch the distributed RAF (create many MPI processes) on user-specified nodes and exchange data for initializing NCCL, while NCCL is responsible for transferring and computing tensors among multiple GPUs efficiently.

```
          RAF IR
             │
             │ Executed by
             ▼
    RAF VM / Interpreter
             │
             │ Dispatch to
             ▼
┌───RAF Op Implementation
│     (in NCCL dialect)
│            │
│            │ Request
│            ▼             Init
│    RAF NCCLCommunicator◄──────RAF MPICommunicator
│            │                            │
│ Call       │ Provide Handle             │ Call
│            ▼                            ▼
└────►NCCL Collective Op          MPI Library APIs


                 Architecture Diagram
```

## Collective Communication Operators

To check the available collective communication operators, we could run the following code. For the detailed
description of each operator, we suggest to refer [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html).

```python
$ python3
>>> import raf
>>> dir(raf.distributed.op)
```
Here we take `AllReduce` as an example to demonstrate how to use it to aggregate an array or a list of arrays. Assume we run RAF with `mpirun` on a cluster consisting of two nodes, two GPUs per node, then the binding will look like,

```
MPI rank 0: cuda(0) @ node 0
MPI rank 1: cuda(1) @ node 0
MPI rank 2: cuda(0) @ node 1
MPI rank 3: cuda(1) @ node 1
```

Like many other multi-GPU MPI applications, a number of RAF processes will spawn, and each process will be assigned with an index `rank` and a GPU in order.

To figure out which GPU this process binds to, we could obtain this information at `DistConfig`.

**Deprecation Notice.** Some attributes in `DistConfig`, including `rank` and `size`, will be deprecated soon, and user should get these from `Communicator` in the future instead.


``` python
import numpy as np
import raf
from raf import distributed as dist

dcfg = dist.get_config()
root_rank = dcfg.root_rank     # The root rank.
size = dcfg.size               # The number of total MPI processes.
rank = dcfg.rank               # The rank of this process, ranging from 0 to (size - 1).
local_rank = dcfg.local_rank   # The local rank of this process on this machine.
local_size = dcfg.local_size   # The number of local MPI processes onthis machine.

device = f'cuda({local_rank})' # the rank-th GPU on this machine that this process binds to.
```

Then we could invoke AllReduce operator. Note that currently AllReduce only supports aggregating tensors located at GPU, because collective operators in MPI dialect are not implemented yet.

``` python
x = np.ones(shape=(1, 4), dtype="float32") * (rank + 1)
x = raf.array(x, device=device)
print("Before allreduce: ", x)
x = raf.allreduce(x)
# Equivalent to: x = raf.allreduce([x], "sum")
# Note that allreduce can accept multiple tensors at a time
print("After allreduce : ", x)
```

In this case, after running the above code on the cluster, the output should be,

| Status | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
| ---    | ---    | ---    | ---    | ---    |
| Before allreduce | [1, 1, ...] | [2, 2, ...] | [3, 3, ...] | [4, 4, ...] |
| After allreduce | [10, 10, ...] | [10, 10, ...] | [10, 10, ...] | [10, 10, ...] |

### Grouping

A portion of RAF collective operators allow users to partition compute resources into several groups analogous to `MPI_Group`. Here is the list of operators supporting this experimental feature.

- `raf.allreduce`
- `raf.allgather`

These operators accept an additional parameter `rank_list`, which is a list of rank subsets, and the process will only talk to ranks in the same subset, and its syntax is straightforward.

Continue with the example above. If we want to let rank 0 talk to rank 1 only, and let rank 2 talk to rank 3, we could simply set the `rank_list` to,

```python
x = raf.allreduce([x], "sum", rank_list=[[0, 1], [2, 3]])
```

To help you better understand this syntax, we provide more examples in the following table for comparison. Note that AR is short for AllReduce.

| Status | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
| ---    | ---    | ---    | ---    | ---    |
| Before allreduce | [1, 1, ...] | [2, 2, ...] | [3, 3, ...] | [4, 4, ...] |
| AR() / AR([[0, 1, 2, 3]]) | [10, 10, ...] | [10, 10, ...] | [10, 10, ...] | [10, 10, ...] |
| AR([[0, 1], [2, 3]]) | [3, 3, ...] | [3, 3, ...] | [7, 7, ...] | [7, 7, ...] |
| AR([[0, 1, 2], [3]]) | [6, 6, ...] | [6, 6, ...] | [6, 6, ...] | [4, 4, ...] |

Note that if a rank doesn't appear in `rank_list`, this rank will run in standalone mode. For instance, `rank_list=[[0, 1, 2]]` is implicitly equivalent to `rank_list=[[0, 1, 2], [3]]`. But it is still suggested to write out all the ranks explicitly. Besides, each rank cannot appear in `rank_list` twice or more, otherwise it will lead to a fatal error.

## Distributed RAF Internals

This section introduces implementation details of distributed RAF. The architecture diagram presented at Section [Overview](#overview) illustrates the essential components that the distribute system comprises, and we will discuss them in the subsections below.

### Distributed IR

There are several RAF passes that help user convert IR for single machine into distributed IR, and we have written articles for each of them. For more details, please move to

- [Auto Data Parallel](./pass/AutoDataParallel.md)
- [Zero Optimizations](../2_user_guide/Distributed-Training.md)

Of course, a correct handwritten distributed IR that manually manipulates data shards is also valid.
### Communicator

The term *Communicator* may refer to various things in different contexts. To disambiguate, we will use the following phrases to make a distinction.

- Communicator / Sub-Communicator / Global Communicator: Refers to RAF Communicator, a TVM object / class that stores some runtime data of the distributed context.
- NCCLCommunicator / MPICommunicator: Refers to a derived class from Communicator.
- NCCL Communicator / MPI Communicator: Refers to an abstract concept defined by NCCL / MPI Library.

NCCL / MPI Communicator establishes connections between workers, provides each machine an identity and important information (e.g., machine's rank, world size, the handler of NCCL / MPI Communicator, etc.), and exchange data among workers. Thus, we consider each NCCL / MPI Communicator defines a distributed context. To support numerous collective communication libraries and manage multiple communicators, we propose a unified data structure *Communicator* to assist them and record the key information including,

- `local_rank`: local rank ID related to physical hardware location, from this communicator's perspective.
- `local_size`: The amount of processes on the same physical machine, from this communicator's perspective.
- `rank`: rank ID of this process from this communicator's perspective.
- `size`: The amount of processes from this communicator's perspective.
- `global_rank`: rank ID of this process from the global communicator's perspective.
- `global_size`: The amount of processes from the global communicator's perspective.
- `group_id`: The ID of group where this process locates. Used by grouping and sub-communicator.
- `group_size`: The number of groups. Used by grouping and sub-communicator.
- Handler. The handler of corresponding MPI / NCCL Communicator.

You might get confused about the three types of `rank`, and why we need multiple MPI / NCCL Communicators. Let us continue the last example we used in Section [Collective Communication Operators](#collective-communication-operators), and explain what global communicators and sub-communicators are.

**Global Communicator.** This communicator will talk to all the ranks specified by the launcher (e.g., `mpirun`), and it is the default communicator when grouping feature is not used.

**Sub-Communicator.** This communicator will only talk to a subset of ranks. It will be automatically created when `rank_list` is specified.

If we execute the code below,

```python
x = raf.allreduce([x], "sum", rank_list=[[1, 2, 3], [0]])
```

then the execution flow will look like this,

- RAF Interpreter / VM dispatches this Op Call to `raf::op::NCCLAllReduce`
- `raf::op::NCCLAllReduce` requests for `raf::NCCLCommunicator([[1, 2, 3], [0]])` sub-communicator.
    - `raf::CommunicatorPool` looks up cache, has not found.
    - `raf::CommunicatorPool` creates and returns a new sub-communicator.
- `raf::op::NCCLAllReduce` gets the handler of NCCL Communicator `ncclComm_t` from `raf::NCCLCommunicator([[1, 2, 3], [0]])` sub-communicator and passes it to `ncclAllReduce` as an function argument.

And at this moment, the attribute values of communicator objects should be,


| Attr | Comm | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
| ---  | ---  | ---    | ---    | ---    | ---    |
| `local_rank/size` | Global | 0/2 | 1/2 | 0/2 | 1/2 |
| | Sub | 0/1 | 0/1 | 0/2 | 1/2 |
| `rank/size` | Global | 0/4 | 1/4 | 2/4 | 3/4 |
| | Sub | 0/1 | 0/3 | 1/3 | 2/3 |
| `world_rank/size` | Global & Sub | 0/4 | 1/4 | 2/4 | 3/4 |
| `group_id/size` | Sub | 1/2 | 0/2 | 0/2 | 0/2 |

```
 ┌─────────────┐   ┌─────────────┐
 │Machine 0    │   │Machine 1    │
 │ ┌─────────┐ │   │ ┌─────────┐ │
 │ │Group 1  │ │   │ │ Group 0 │ │
 │ │ ┌─────┐ │ │   │ │ ┌─────┐ │ │
 │ │ │GPU 0│ │ │   │ │ │GPU 2│ │ │
 │ │ └─────┘ │ │   │ │ └─────┘ │ │
 │ │         │ │   │ │         │ │
 │ └─────────┘ │   │ │         │ │
 │             │   │ │         │ │
 │ ┌───────────┼───┼─┘         │ │
 │ │           │   │           │ │
 │ │ ┌─────┐   │   │   ┌─────┐ │ │
 │ │ │GPU 1│   │   │   │GPU 3│ │ │
 │ │ └─────┘   │   │   └─────┘ │ │
 │ │           │   │           │ │
 │ └───────────┼───┼───────────┘ │
 │             │   │             │
 └─────────────┘   └─────────────┘

      Group([[1, 2, 3], [0]])
```

**Obtain Communicators.** There are two ways to obtain communicators.

- `RequestDistributed()`
- `CommunicatorManager::Get()`

It is recommended for communication operators to invoke `RequestDistributed()`. Although currently there is not a significant difference between them, we might apply some optimization strategies to the first method in the future.

**Available Communicators.**

| Type | Has Op Impl | Grouping Support | Purpose |
| ---  | ---         | ---              | ---     |
| Void | No | Yes | Dummy for test purposes and standalone mode |
| MPI | No | No | Help NCCL to initialize |
| NCCL | Yes | Yes | Exchange data between GPUs efficiently |