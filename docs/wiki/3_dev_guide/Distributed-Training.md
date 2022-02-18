<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Distribution Mechanism

This article explains the distribution mechanism in Meta.
For the tutorial of distributed training, please see [here](../2_user_guide/Distributed-Training.md)

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


### Data Parallelism

WIP

### ZeRO Optimizations

WIP

### Possible Improvements

* Fuse small tensors to reduce communication overhead.
