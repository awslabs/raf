<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Distributed Training

This tutorial introduces how to train your model with multiple GPUs.
To enable distributed training, you need to turn `RAF_USE_CUDA`, `RAF_USE_MPI`, and `RAF_USE_NCCL` on in `${RAF_HOME}/build/config.cmake` before cmake.

For implementation details of collective communication operators, data parallel and ZeRO optimizations, please see [Distribution Mechanism](../3_dev_guide/Distribution-Mechanism.md).

## Enable Distributed Training Environment

To enable distributed training, you need to set the corresponding flags in the distributed config. For example:

```python
import raf
from raf import distributed as dist
dcfg = dist.get_config()
dcfg.enable_data_parallel = True
```

Note that if you are using the provided script (i.e., `dist_example.py`), you can simply change the values in `raf_dist_config`. We will introduce each configure in the following subsections along with the distribution methodologies.

MPI is recommended to manage multi-processing, so we need to launch the script with `mpirun`:

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

To use other launchers, see [Distribution Mechanism](../3_dev_guide/Distribution-Mechanism.md).

### Data Parallelism

Data parallelism distributes the input training data to each device, and performs
`AllReduce` on gradients.

To enable data parallelism, set the corresponding configure to be `True` in the script:

```python
raf_dist_config = {
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
raf_dist_config = {
    "zero_opt_level": 1, # Use ZeRO-1
    ...
}
```

