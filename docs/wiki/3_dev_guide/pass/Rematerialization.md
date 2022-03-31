<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Rematerialization

This article aims to provide an intuitive understanding of the rematerialization pass in RAF. For more details please refer to the source code at `src/pass/rematerialization.cc`. 

## What is rematerialization?

When training large models, it is common that the total size of model parameters and activations exceeds the memory capacity of the GPU. In such cases, one solution is to reduce the batch size so that the total size of activations can be reduced. However, using a small batch size often results in lower training throughput, and may hurt the convergence rate in some cases. 

Rematerialization (also referred as gradient/activation checkpointing in other frameworks and literatures) is another solution to enable larger training batch size on machines with limited GPU memory. The key idea behind rematerialization is to trade compute for memory: if we run out of GPU memory at a certain point of execution, we can remove some tensors from the GPU memory to get some free space, and recompute those tensors later when we need them. This idea is better illustrated with an example:

```
/* Suppose the size of each tensor is one unit */
/* Our memory budget is three units */
/* We assume that input arguments cannot be freed */
fn(input_0) {
    let x0 = add(input_0, input_0);
    let x1 = mul(x0, input_0);
    /* Out of memory here, cannot allocate space for x2 */
    let x2 = add(x1, input_0); 
    let x3 = div(x0, x2);
    x3
}
```

In the example above, we run out of memory when we try to compute `x2`: `input_0`, `x0`, and `x1` are inside the GPU memory and they have used all of the memory budget. We need four units of memory to complete the computation of this function. With rematerialization, we can modify the IR as follows:

```
fn_with_remat(input_0) {
    let x0 = add(input_0, input_0);
    let x1 = mul(x0, input_0);
    /* x0 can be freed here, it won't be used in the future */
    /* Now we have one unit of memory for x2 */
    let x2 = add(x1, input_0); 
    /* Rematerialize x0 as x0_remat before computing x3 */
    let x0_remat = add(input_0, input_0);
    /* Use x0_remat instead of the original x0 here */
    let x3 = div(x0_remat, x2);
    x3
}
```

Notice how `x0` is rematerialized as `x0_remat` before it is used to compute `x3`. With proper memory allocation, we can free `x0` after computing `x1`, thus leaving one unit of memory for `x2`. The peak memory is reduced from four units to three units at the cost of one more addition (`let x0_remat = add(input_0, input_0)`). In practice this saved peak memory can help us train models with larger batch sizes. 

## How to enable it in RAF?

The rematerialization pass in RAF takes two parameters:
- `raf.memory_budget`: The GPU memory budget in bytes. Setting this parameter to zero disables the rematerialization pass. 
- `raf.remat.use_gflops_cost`: Set this parameter to `True` to use a GFLOPS-based operator cost function instead of the default profiling-based cost function. The GFLOPS-based cost function is faster to run, but is also less accurate. 

Please specify these parameters in the `PassContext` to enable the rematerialization pass. An example of enabling the rematerialization pass on a single GPU with 16GB memory would be:
```
with raf.ir.PassContext(
    config={
        # Leave some space for CUDA runtime, etc. 
        "raf.memory_budget": 13 * (1024 ** 3), 
        "raf.remat.use_gflops_cost": False,
    }
):
    # Run your model
    ...
```

Notice that when the rematerialization pass is enabled, it may still fail when the memory budget is too tight. In such case, you will have to either relax the memory budget or reduce the training batch size. 

## How much does it help?

The rematerialization pass allows training using 2x or even larger batch size without significant throughput degradation. Some results on popular language models are as follows:

| Model | Maximum non-remat batch size | Remat batch size | Throughput relative to non-remat |
|--|--|--|--|
| BERT-base-mlm  | 16 | 32 | 94.3% |
| BERT-large-mlm | 8  | 16 | 96.5% |
| GPT2           | 8  | 24 | 93.2% |

## How does it work?

This section dives into a bit more details of the RAF rematerialization pass. If you are interested in the rematerialization problem, check the following papers as a starting point. The RAF rematerialization pass combines the ideas from the state-of-the-art. Please feel free to reach out to us if you have ideas on how to further improve this pass. 
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174.pdf)
- Jain et al., [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://arxiv.org/pdf/1910.02653.pdf)
- Kirisame et al., [Dynamic Tensor Rematerialization](https://arxiv.org/pdf/2006.09616.pdf)

The goal of the rematerialization pass is to keep the memory usage at any point of training under the provided memory budget, while minimizing the computational overhead caused by rematerializing tensors. While this problem can be formulated as an Integer Linear Programming (ILP) problem and solved exactly, solving the ILP can be pretty time-consuming. As a result, we develop a fast heuristic to find a good rematerialization strategy at compile time. A sketch of our algorithm is as follows:

```
memory consumption = total size of model parameters
foreach operator in RAF IR:
    update memory consumption to reflect memory utilization at this operator
    if memory consumption > budget:
        while memory consumption > budget:
            choose the lowest-score tensor, T, from all live tensors
            free T from memory
            memory consumption -= sizeof(T) 
```

The score of each tensor is computed according to its size and the total latency required to recompute it. The latency of each operator is obtained through profiling by default. As mentioned earlier, you can set `"raf.remat.use_gflops_cost" = False` to use the faster but less-accurate GFLOPS-baed cost instead. 