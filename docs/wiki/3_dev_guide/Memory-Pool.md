<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Memory Pool

This document introduces the Memory Pool of RAF.

## Strategies

Currently, there are two types of memory pool in RAF: 

1. **Page Unit Pool.** A general concept of page unit pool is reusing the allocated memory as possible. Specifically, page unit pool holds a shared pointer of each allocated memory buffer. When user requests a memory buffer, and the page unit pool has a buffer with the requested size that is not being used, then page unit pool simply returns the shared pointer instead of allocating a new buffer. In addition, to reduce the fragmentation, the size of each memory request is rounded up to a page unit (e.g., assuming the page size is 4KBs, then a request of 3KBs will still get a 4KB buffer), so that the requests result in the same size could potential share the buffer.

2. **No Pool.** As its name indicates, this memory pool does not maintain a "pool". All requests of allocating or freeing memory are directly proceed by the device APIs, and result in significant latency overheads.

The strategy of adopting memory pool is described as follows. By default, we use page unit pool for both CPUs and GPUs, which could bring down the running time by almost 50% for ResNet-50, VGG and other models compared with no pool.

On the other hand, since CUDA 11.2, CUDA has a builtin memory pool [[1]](https://developer.nvidia.com/blog/enhancing-memory-allocation-with-new-cuda-11-2-features/). Similar to page unit pool, CUDA memory pool also holds the allocated memory for a process, meaning that `cudaFreeAsync` just marks the memory as free instead of returning to the device until the process is terminated or the synchronization API is called, so the memory still belongs to the current process and can be directly used when `cudaMallocAsync` is called later. Note that CUDA memory pool is relateively mature in CUDA 11.3, so we choose no pool when CUDA version is later than 11.3 to directly leverage the CUDA memory pool.

We perform an experiement to illustrate the memory pools. The experiment was done with one layer BERT model with batch size 4 and sequence length 128 on Tesla T4 GPU. The following table shows the total latency of requesting membry buffers in the first and second iteration.

Strategy | 1st iteration | 2nd iteration
--|--|--
No Pool (CUDA 11.0)	|25.2ms	|19.3ms
Page Unit Pool (CUDA 11.0)	|15.2ms	|0.9ms
No Pool (CUDA 11.3)	|14.9ms	|1.7ms

As can be seen, the allocation latency of page unit pool and CUDA memory pool is about 50% shorter than no pool in the first iteration, and is even 90% shorter in the second iteration. The result implies that 1) about 50% memory buffers can be reused within the same iteration, and 2) almost all memory buffers can be reused cross iterations.

## Change strategy

If you want to use no_pool, you can change it through Python API `InitPool(device, pool_name)`. Here is an example:

``` python
# Changing to no_pool strategy
import raf
from raf._ffi.memory_pool import InitPool, RemovePool
from raf._core.core_utils import str2dev

InitPool(str2dev("cuda"), "no_pool")

# Run the model.
...
```

If you want to change back to default memorpy strategy, you can call `RemovePool(device)` or `InitPool(device, "page_unit_pool")`. Note that everytime you call `InitPool`, the current pool will be removed first, even if the new pool's name is equal to the current one. As a result, if you change the memory pool in the middle, the new memory pool will lose the buffer pointers of already allocated ndarrays and may result in memory leak.

## Design a new memory pool

If you want to develop your own memory pool, you can follow the following instructions.

### Step 1: Prepare

You should create a new folder under $RAF_HOME/src/memory_pool, and create a new cpp file that named as same to the folder name (recommended). The recommended name should be like `xxx_pool`.

### Step 2: Implement your pool

To begin, you need include `"raf/device_api.h"`,`"raf/memory_pool.h"`, `"raf/registry.h"`, and wrapper your code with namespace `raf->memory_pool->your_pool`.
You will first need a memory wrapper that holds the actual memory. It must derived from `raf::memory_pool::Memory`.

Then you can create the Pool Class that derived from `raf::memory_pool::MemoryPool`.

### Step 3: Register your pool

Remember to register your pool in the cpp file you created, the code should be like:
`RAF_REGISTER_GLOBAL("raf.memory_pool._make.your_pool").set_body_typed(YourPool::make);`

After re-make RAF, you can enable your pool by calling `InitPool(contxt, pool_name)`.
