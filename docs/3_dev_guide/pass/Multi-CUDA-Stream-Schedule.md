
<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Multi CUDA-Stream Schedule

This article aims to provide an overview of CUDA stream support and multi-stream schedules in RAF. CUDA multi-stream is capable of improving the parallelization of executing a model with branches in its architecture. For example, with CUDA multi-stream support, we are able to achieve 1.2x to 1.3x speedup for single batch inference of Inception V3 on NVIDIA GPUs (e.g., Tesla V100 and GeForce RTX 2080 Ti).

## Background

We first briefly introduce CUDA streams and CUDA events to illustrate the motivation of supporting CUDA mutli-streaming in RAF.

### CUDA Stream

In CUDA[1], a stream[2] is a sequence of tasks that will be executed in order. A task can be a kernel, memory copy, or a host callback function. On the other hand, we can create multiple streams, and the tasks on different streams will be executed out of order concurrently. 

The following tasks can be executed concurrently if they are on different streams (for full list, please refer to CUDA Programming guide[1]):
- Kernel launches
- Memory copy between host and device

There are implicit synchronizations between two tasks on different streams if one of the following tasks is executed between the two tasks:
- Memory allocation (device memory or page-locked host memory)
- Device memory set
- Device memory copy (of the same device)
- Any task on default stream (also called NULL stream, please refer to [2] for the behaviors of default stream)
- The change of L1/shared memory configuration

### CUDA Event

CUDA provides CUDA Event to specify the dependency and synchronization between different streams. We can record a CUDA event on a stream A and wait this event on another stream B. All subsequent tasks on stream B would wait for the completion of preceding tasks of the event. 

### Purposes of CUDA Stream in RAF

CUDA stream provides a mechanism to issues tasks without dependency. With CUDA stream, we can transfer data from GPU A to GPU B while executing a compute-heavy kernel on GPU A. Besides this, we can also launch multiple small kernels to better utilize the GPU. This article will mainly discuss the later one. 

In CUDA, we launch a kernel by issuing a grid of thread blocks to the GPU. A grid contains a group of thread blocks, while each thread block contains a group of threads. Each thread block will be placed on a single Stream-Multiprocessor (SM) unit of the GPU, while each SM may contains multiple thread blocks. When we launch a small kernel (e.g., with a small number of thread blocks), the device will suffer from the under-utilization problem because a large number of SMs have not been utilized. Taking a convolution with batch size 1, input channels 375, output channels 750, kernel size 3x3, strides 1x1, and input image size 15x15 as an example, the cuDNN[3] library will launch a kernel with 48 thread blocks on RTX 3070 Laptop. RTX 3070 Laptop has 40 SMs, while each SM on this GPU can run two such thread blocks. Thus, 32/80 on-chip storage (i.e., shared memory and registers) and computation units are wasted when running this kernel by roughly estimation. 

To alleviate this problem, we can launch dependency-free kernels on different streams, which allows the GPU to place multiple kernels on the device to use the under-utilized on-chip resources.

## Stream Related Operators and Instructions in RAF

In RAF, we added several operators in graph-level IR as well as several instructions in underlying virtual machine to support CUDA stream and events. In the graph-level IR, we have added the following operators:
- `set_stream(int device_id, int stream_id)`: Set the current stream to the given stream on given device.
- `add_event(int event_id, int stream_id = -1)`: Add an event to given stream.
- `wait_event(int event_id, int stream_id = -1)`: Make given stream wait for given event.

The corresponding VM instructions:
- `CudaSetStream device_id, stream_id`
- `CudaAddEvent event_id, stream_id`
- `CudaSetEvent event_id, stream_id`

These operators only make sense in A-normal form, in which the execution order is naturally specified. Meanwhile, it also implies that we have to be careful when manipulating the IR with multi-stream ops for other optimizations.


## Multi-Stream Schedules

There are three scheduling algorithms have been implemented in RAF: wavefront schedule, as-soon-as-possible (ASAP) schedule, and inter-operator-scheduler (IOS) schedule[4]. Each schedule is implemented as a compiler pass, which transform the computation graph from graph normal form (GNF) to A-normal form (ANF) and injecting the scheduling operators. The difference between these schedule passes are their strategy of parallelization. At most one schedule pass will be used and can be configured by `raf.stream_schedule.policy` in PassContext like the following code
```python
with raf.ir.PassContext(opt_level=2, config={"raf.stream_schedule.policy": "wavefront"}):
    ...
```

Please refer to `src/pass/stream_schedule_{wavefront/asap/ios}.cc` for the implementation details, and refer to `tests/python/pass/test_pass_stream_schedule_{wavefront/asap/ios}.py` for the usage. 

#### Wavefront Schedule

The wavefront schedule repeats the following steps to partition the computation graph into waves:
```python
waves = []
while graph not empty:
    nodes = [all operators in the graph with zero degree]
    chains = []
    for node in nodes:
        chain = []
        walk along the node to its succeeding operators, 
            adding the operator to chain, stop when we meet an op with multiple succeeding operators
        append chain to chains
    append chains to waves
    delete all nodes in chains from graph
```
Taking the following computation graph as an example, we can get a schedule with 4 waves.

![Example](https://user-images.githubusercontent.com/23381083/122089070-fd69ec80-ce38-11eb-981c-4947626283f4.png)

The wave will be executed one by one, and different chains in a wave will be placed on different streams.

#### ASAP Schedule

The ASAP schedule takes the following steps to construct the schedule:

1. Classify all edges in the computation graph into two categories: heavy edge and
   light edge. 
   Each node can have at most one heavy edge points to it and at most one heavy
   edge points out from it. We call the node it depends on through a heavy edge heavy child,
   and the node that depends on it heavy parent. Other nodes it depends on are light children
   and other nodes that depend on it light parents. After the edge classification, we partition
   the dependency graph into multiple heavy chains that connected by heavy edges.

2. We would launch each heavy chain on a cuda stream. Let A and B be two heavy chains. If all
   nodes in A are before all nodes in B in dependency graph (i.e., for all a in A and b in B,
   there is a path from a to b), we may reuse the stream used to launch heavy chain A as the
   cuda stream to launch heavy chain B. For each node that has light parents, we will add an
   event for that node. Before we launch a node with light children, we will wait for the
   events of light children.

#### IOS Schedule

IOS schedule pass utilizes a dynamic algorithm to search the schedule. It partitions the computation graph into different stages and use profiler to measure the latency of different stages. It uses dynamic programming to avoid the recomputation of subgraphs during partitioning the computation graph. Please refers to the paper[4] for more information about this algorithm.

## Related Work
Please refer to the following works if you want to know more about the inter-operator scheduling in deep neural networks:
- Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks [5]
- IOS: Inter-Operator Scheduler for CNN Acceleration [4]
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning [6]

## Reference
- [1] CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.
- [2] Section 3.2.6 'Asynchronous Concurrent Execution' in CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
- [3] cuDNN: https://developer.nvidia.com/cudnn
- [4] IOS: https://proceedings.mlsys.org/paper/2021/hash/38b3eff8baf56627478ec76a704e9b52-Abstract.html 
- [5] Rammer: https://www.usenix.org/conference/osdi20/presentation/ma
- [6] Nimble: https://papers.nips.cc/paper/2020/hash/5f0ad4db43d8723d18169b2e4817a160-Abstract.html
