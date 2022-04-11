<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Operator Dialects

RAF supports many backend implementations, such as Apache TVM, CuBLAS, CuDNN, CUTLASS, and manual written CUDA kernels. Each backend performs better in certain types of operators and dtypes while worse in others, and they have different fusion capabilities. To better manage all backends and use the best operators, RAF introduces a systematic approach, operator dialects, to perform operator disaptching.

This article covers 1) the basic concept of dialect ops, 2) dialect op fusion, and 3) how to test dialect ops.

## Base Ops and Dialect Ops

In RAF, we categorize ops to two types -- base ops and dialect ops. Base ops are named to `raf.op.name`. RAF works on the graph IR with base ops in most optimization process. Meanwhile, since base ops are backend independent, they have to be changed to a particular dialect op before kernel compilation and execution.

On the other hand, dialect ops are named to `raf.op.dialect.name` and are backend specific. For example, `raf.op.tvm.softmax` indicates that this `softmax` op will be lowered to Apache TVM backend to generate an executable binary; `raf.op.cudnn.conv2d` indicates that this `Conv2D` op will be dispatched to CuDNN library.

There are two methods to change base ops to dialect ops in RAF, depending on which runtime user is using. In the case of interpreter which directly traverses the IR and JITs each call node of base op one by one, the op dispatcher simply follows the predefined dialect op priority level to determine which dialect op should be used. For example, the priority of `MatMul` op is shown as follows:

```
RAF_REGISTER_DIALECT_OP(cublas, matmul, 15);
RAF_TVM(matmul, ...); // TVM level is always 10
```

It means the dispatch order of `MatMul` is CuBLAS > TVM. This order has some interesting meanings:

1. If CuBLAS is available (e.g., the target device is GPU and CuBLAS is enabled), then use CuBLAS.
2. Otherwise (e.g., the target device is CPU or CuBLAS is disabled), then use TVM. Since TVM is the primary compiler backend in RAF and is supposed to be robust, we expect it to be available all the time.

On other other hand, the second method to change base ops to dialect ops is via the fusion passes, which will be introduced in the next section.

## Dialect Op Fusion

As mentioned in the beginning of this article, one important reason of introducing dialect ops is to cover different fusion capabilities. In RAF, we have two fusion strategies.

1. For all backends except for TVM, we expect the backends are either vendor libraries such as CuDNN, or kernel templates such as CUTLASS. Since their fusion capability is limited, we leverage pattern matching to fuse them. Taking the `MatMul` again as an exemple, we register the following dialect pattern:

    ```python
    def _cutlass_matmul_fusion(matmul_ops, dtype=None):
        act_ops = ["raf.op.relu", "raf.op.gelu"]
        # matmul
        matmul = call_binary_ops(matmul_ops, dtype)
        # bias
        bias = wildcard()
        beta = has_shape(()) | has_shape((1,))
        scaled_bias = is_op("raf.op.multiply")(beta, bias)
        bias = scaled_bias | bias
        # pattern: matmul+scaled_bias or matmul+bias
        with_bias = is_op("raf.op.add")(matmul, bias, *n_null_constant(2))
        # pattern: matmul+(scaled_)bias+act or matmul+act
        with_act = is_ops(act_ops)(with_bias | matmul)
        # We exclude the single matmul op pattern as ther perf of cutlass is worse than cublas
        return with_act | with_bias

    register_pattern(_cutlass_matmul_fusion(MATMUL_OPS), "cutlass", 10, "matmul_fusion")
    ```

    The above pattern will be examined by a RAF pass `FuseDialect`. Once the pattern is matched, the matched subgraph will be fused to a function and makred as CUTLASS:

    ```
    %8 = fn (%p, %p9, %p10, Primitive=1, Dialect="cutlass", PatternName="matmul_fusion") {
        %7 = raf.op.cutlass.matmul(%p, %p9);
        raf.op.cutlass.add(%7, %p10)
    };
    ```

    Consequenly, the above fused function will be dispatched to CUTLASS backend when JITing.

2. For TVM backend, since TVM is capable of performing code generation, it supports very flexible fusion strategies. For example, we could fuse any types and numbers of elementwise ops to a reduction op. Ideally, the fused elementwise ops will be inlined to the reduction op and become a free lunch. As a result, the fusion strategy for TVM backend is rule based. We only assign a fusion pattern (e.g., elementwise, broadcast, injective, common reduce, etc) to each TVM dialect op. The RAF pass `FuseTVM` will be applied after `FuseDialect` and fuse all rest base ops as many as possible. Here is an example:

    ```
    %85 = fn (%p019, %p121, %p219, %p316, %p413, %p63, %p73, %p82, %p91, %p101, %p1110, %p122,
              %p131, %p141, %p151, %p161, %p171, %p181, %p191, %p201, Primitive=1, Dialect="tvm") {
        %67 = raf.op.tvm.reshape(%p219, %p316);
        %68 = raf.op.tvm.cast(%67, %p413);
        %69 = raf.op.tvm.multiply(%68, %p59);
        %70 = raf.op.tvm.tanh_dx(%p019, %p121, %69);
        %71 = raf.op.tvm.multiply(%70, %p63);
        %72 = raf.op.tvm.cast(%p91, %p101);
        %73 = raf.op.tvm.power(%72, %p82);
        %74 = raf.op.tvm.cast(%73, %p1110);
        %75 = raf.op.tvm.divide(%74, %p91);
        %76 = raf.op.tvm.cast(%75, %p122);
        %77 = raf.op.tvm.multiply(%71, %p73);
        %78 = raf.op.tvm.multiply(%p82, %76);
        %79 = raf.op.tvm.multiply(%77, %78);
        %80 = raf.op.tvm.cast(%67, %p151);
        %81 = raf.op.tvm.multiply(%80, %p161);
        %82 = raf.op.tvm.add(%71, %79, %p131, %p141);
        %83 = raf.op.tvm.multiply(%81, %p171);
        %84 = raf.op.tvm.add(%82, %83, %p181, %p191);
        raf.op.tvm.reshape(%84, %p201)
    };
    ```

## Test Dialect Op

Finally, we introduce an RAF utility that could help test (single) dialect op. Here is an example:

```python
import time

import raf
from raf.testing import randn, with_dialect

n, k, m = 8, 4, 16
device = "cuda"
dtype = "float32"

m_a, _ = randn((n, k), device=device, dtype=dtype)
m_b, _ = randn((k, m), device=device, dtype=dtype)

@with_dialect(["tvm"])
def run_tvm():
    return raf.matmul(m_a, m_b)

# Warmup
for _ in range(10):
    run_tvm()

start = time.time()
run_tvm()
print("TVM (us):", 1e6 * (time.time() - start))

@with_dialect(["cublas"])
def run_cublas():
    return raf.matmul(m_a, m_b)

# Warmup
for _ in range(10):
    run_cublas()

start = time.time()
run_cublas()
print("CuBLAS (us):", 1e6 * (time.time() - start))
```

Output:

```
TVM (us): 655.4126739501953
CuBLAS (us): 438.690185546875
```

The decorator `with_dialect` enforces the dialects to be dispatched for all ops in this function. Note that we do not tune the TVM matmul op, so the performance is much worse than CuBLAS in this example.
