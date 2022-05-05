<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Automatic Data Parallelism

This article aims to explain how RAF does data parallel automatically and how to enable it. In short, RAF accepts single device IR from user models, and applies the AutoDataParallel Pass to generate Data Parallel IR.

## How does it work?

AutoDataParallel Pass is applied on a training graph, meaning that enables data parallelism after the auto-differentiation by automatically transforming the backward closure in training graph. The transformations include:
1. Add communication operator (e.g., `all_reduce`) after each operator generates local gradient.
2. Replace the returned local gradient by the corresponding aggregated global gradient.

For example, here we have the following forward graph:

```
def @main(%x: Tensor[(2, 2), float32], %y_true: Tensor[(2), int64], %c: Tensor[(2, 2), float32]) {
  let %a1 = raf.op.matmul(%x, %c);
  let %a2 = raf.op.nll_loss(%y_true, %a1);
  %a2
}
```

After AutoDiff pass, the graph becomes:

```
def @main(%x: Tensor[(2, 2), float32], %y_true: Tensor[(2), int64], %c: Tensor[(2, 2), float32]) -> (Tensor[(1), float32], fn (Tensor[(1), float32]) -> (Tensor[(2, 2), float32], Tensor[(2), int64], Tensor[(2, 2), float32])) {
  let %a1 = raf.op.matmul(%x, %c) /* ty=Tensor[(2, 2), float32] */;
  let %a2 = raf.op.nll_loss(%y_true, %a1) /* ty=Tensor[(1), float32] */;
  let %adjoint_closure = fn (%dy: Tensor[(1), float32]) -> (Tensor[(2, 2), float32], Tensor[(2), int64], Tensor[(2, 2), float32]) {
    let %x_0 = raf.op.nll_loss_dpred(%dy, %y_true, %a1) /* ty=Tensor[(2, 2), float32] */;
    let %x_1 = raf.op.matmul_nt(%x_0, %c) /* ty=Tensor[(2, 2), float32] */;
    let %x_2 = raf.op.matmul_tn(%x, %x_0) /* ty=Tensor[(2, 2), float32] */;
    let %x_3 = raf.op.zeros_like(%y_true) /* ty=Tensor[(2), int64] */;
    let %x_5 = (%x_1, %x_3, %x_2);
    %x_5
  };
  let %ret = (%a2, %adjoint_closure);
  %ret
}
```

After applying AutoDataParallel pass the graph is:

```
def @main(%x: Tensor[(2, 2), float32], %y_true: Tensor[(2), int64], %c: Tensor[(2, 2), float32]) -> (Tensor[(1), float32], fn (Tensor[(1), float32]) -> (Tensor[(2, 2), float32], Tensor[(2), int64], Tensor[(2, 2), float32])) {
  let %a1 = raf.op.matmul(%x, %c) /* ty=Tensor[(2, 2), float32] */;
  let %a2 = raf.op.nll_loss(%y_true, %a1) /* ty=Tensor[(1), float32] */;
  let %adjoint_closure = fn (%dy: Tensor[(1), float32]) -> (Tensor[(2, 2), float32], Tensor[(2), int64], Tensor[(2, 2), float32]) {
    let %x_0 = raf.op.nll_loss_dpred(%dy, %y_true, %a1) /* ty=Tensor[(2, 2), float32] */;
    let %x_1 = raf.op.matmul_nt(%x_0, %c) /* ty=Tensor[(2, 2), float32] */;
    let %allreduce_in = (%x_1,);
    let %g = raf.op._allreduce(%allreduce_in, str"avg", nullptr) /* ty=Tensor[(2, 2), float32] */;
    let %x_2 = raf.op.matmul_tn(%x, %x_0) /* ty=Tensor[(2, 2), float32] */;
    let %allreduce_in1 = (%x_2,);
    let %g1 = raf.op._allreduce(%allreduce_in1, str"avg", nullptr) /* ty=Tensor[(2, 2), float32] */;
    let %x_3 = raf.op.zeros_like(%y_true) /* ty=Tensor[(2), int64] */;
    let %allreduce_in2 = (%x_3,);
    let %g2 = raf.op._allreduce(%allreduce_in2, str"avg", nullptr) /* ty=Tensor[(2), int64] */;
    let %x_5 = (%g, %g2, %g1);
    %x_5
  };
  let %ret = (%a2, %adjoint_closure);
  %ret
}
```

In the AutoDataParallel pass, NCCL version will be checked. If the NCCL version is 2.10 or above, the average allreduce is to be inserted, just like the example shows above. Otherwise, a sum allreduce is used and then a divide operator is followed, because NCCL version <2.10 does not support average allreduce, we have to perform a division.

## How to enable it in RAF?
Before triggering AutoDataParallel pass, we need to enable it first. Then we can manually apply the pass:

``` python
raf.distributed.get_config().enable_data_parallel = True
record = model._internal(*args, **kwargs)
mod = record.mod

passes = [
    raf._ffi.pass_.InferType(),
    raf._ffi.pass_.AutoDataParallel(),
    raf._ffi.pass_.InferType()
]
seq = raf.ir.RAFSequential(passes)
mod = seq(mod)
```

Another way to triger the AutoDataParallel pass is just invoking: `with_data_parallel`:

```python
raf.distributed.get_config().enable_data_parallel = True
model = raf.optim.data_parallel.with_data_parallel(model)
```

## Run
Data parallel training assumes using multiple devices. Currently, RAF uses MPI to organize the different training processes. If we want to run two training processes the command should look like:

```bash
mpirun -np 2 python3 python_file

```
