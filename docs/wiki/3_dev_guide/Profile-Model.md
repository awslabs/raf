<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Profile Your Model

In this article, we introduce useful profiling tools in RAF to let you monitor the execution overhead and memory consumption in run time.

The latency and memory profiler in RAF are implemented as global status. It means you only need to add a few line in your Python code to make the profiling happen. In this article, we use BERT-base as an example to illustrate how profiler works.

```python
import json
import raf
from raf.testing import get_transformer_model, append_loss_n_optimizer
from raf.testing import randn_torch, one_hot_torch
from raf.testing import run_vm_model, get_vm_executor, run_vm_executor

def setup_model():
  # Load a transformer model. Note that transformers package has to be available.
  model, shape = get_transformer_model(
    "bert-base-uncased", batch_size=32, seq_length=128, dtype="float32"
  )

  # Generate dummy inputs.
  r_x, _ = randint((32, 128), low=0, high=10000, dtype="int64")
  batch = shape[0]
  vocab_size = shape[-1]
  if len(shape) == 3:
      batch = shape[0] * shape[1]
  r_ytrue, _ = one_hot_torch(batch, vocab_size)
  r_dy, _ = randn_torch((), dtype="float32")

  # Append loss and optimizer.
  optimizer = append_loss_n_optimizer(model, [r_x], shape, r_ytrue)
  return optimizer, [r_dy, r_x, r_ytrue]

# Setup model.
optimizer, args = setup_model()

# Initialize a VM executor and run the model.
run_vm_model(optimizer, "cuda", args)
```

## Profile Latency

To profile latency of each operator execution, we simply wrap the model execution with the RAF profiler APIs:

```python
# Setup model.
optimizer, args = setup_model()

# Initialize a VM executor once.
record = optimizer._internal(*args)
executor = get_vm_executor(record.mod, "cuda")

# Run a few time to warmup to exclude the JITing overheads.
for _ in range(10):
    run_vm_executor(executor, record, args, "cuda")

# Clean the existing profiled stats and enable latency profiler.
raf.utils.profiler.clear()
raf.utils.profiler.start()

# Run the model with profiler.
run_vm_executor(executor, record, args, "cuda")

# Disable latency profiler.
raf.utils.profiler.stop()

result = raf.utils.profiler.get()
with open("profile.json", "w") as filep:
    json.dump(result, filep, indent=4)
raf.utils.profiler.clear()
```

Note that the API `raf.utils.profiler.get()` will collect all the profiling results and keep the cache for potential access. Call the API `raf.utils.profiler.clear()` explicitly to clear the cache. The profiling results will be stored in memory in JSON format. You can either save it to a JSON file or directly call `raf.utils.profiler.dump(filename:str)`. Here is a snippet of the dumped JSON log. In short, each executed op will have two entries in the log with the start and end timestamps, and other information.

```
{
    "traceEvents": [
        {
            "name": "raf_op_tvm_cast_76",
            "cat": "Default Stream",
            "ph": "B",
            "ts": 1648086040982992,
            "args": {
                "args_string": "T<768xf32>,|T<768xf16>"
            },
            "pid": 2364,
            "tid": "Default Stream"
        },
        {
            "name": "raf_op_tvm_cast_76",
            "cat": "Default Stream",
            "ph": "E",
            "ts": 1648086040983008,
            "args": {
                "args_string": "T<768xf32>,|T<768xf16>"
            },
            "pid": 2364,
            "tid": "Default Stream"
        },
        {
            "name": "raf_op_tvm_cast_77",
            "cat": "Default Stream",
            "ph": "B",
            "ts": 1648086040983026,
            "args": {
                "args_string": "T<768x768xf32>,|T<768x768xf16>"
            },
            "pid": 2364,
            "tid": "Default Stream"
        },
        {
            "name": "raf_op_tvm_cast_77",
            "cat": "Default Stream",
            "ph": "E",
            "ts": 1648086040983039,
            "args": {
                "args_string": "T<768x768xf32>,|T<768x768xf16>"
            },
            "pid": 2364,
            "tid": "Default Stream"
        },
        ...
    ],
    "displayTimeUnit": "ms"
}
```

### Visualization

Open the file that stores the profiling results with chrome:://tracing.
(Tips: open your chrome, and type "chrome::tracing", then drag your file into it.)

### Profiling Levels

The RAF latency profiler provides 2 levels of profiling. The default level is 1 and it can be configured by `raf.utils.profiler.start(prof_level=?)`.

- Level 1 profiles the kernel execution only. This is used to investigate the performance bottleneck caused by a certain operator.
- Level 2 profiles the kernel execution as well as the VM execution. This is used to investigate the VM overheads during the exection.

### Profile more

If you want profile more content in the backend, you can add your own profiling code following the followed instructions.

#### Profile a CPU function

1. Add `#include "raf/profiler.h"` to the source file with the function you want to profile.
2. Wrap the code snippet with Profiler macros. We provide a macro to profile your code conviently. To use it, you just need to wrap your code into this marco. Example:

    ``` cpp
    // The code snippet you want to profile
    for (int i = 0, n = req->memory.size(); i < n; ++i) {
      RequestMemory(req.get(), i);
    }
    ```

    With profiler:

    ``` cpp
    // Using BASE_PROFLIER to profile, the code snippet is the last argument of this macro.
    WITH_BASE_PROFILER(call->dev, op->name, "MemoryRequest",
                  {"Count: " + std::to_string(req->memory.size())}, {
                    for (int i = 0, n = req->memory.size(); i < n; ++i) {
                      RequestMemory(req.get(), i);
                    }
                  });
    ```

#### Profile a GPU kernel

1. Add `#include "raf/src/profiler/cuda/cuda_profiler.h"` to the source file with the function you want to profile.
2. Wrap the code snippet with CUDA Profiler macros. Just like the base profiler, we also provide a macro `WITH_CUDA_PROFILER` for profiling on cuda asynchronous execution. The usage is same to `WITH_BASE_PROFILER`.

## Profile Op Latency

In addition to the end-to-end latency profiler introduced above, RAF also offers an op profiler that quickly profiles an IR expression. For example:

```python
import raf
from raf._ffi.op_profiler import Profile, ResetCache, GetCacheSize
from raf.testing import run_infer_type

data = raf.ir.var("x", shape=(16, 3, 224, 224))
weight = raf.ir.var("w", shape=(3, 3, 5, 5))
expr = raf.ir.op.conv2d(data, weight)
expr = run_infer_type(expr).body

device = raf.Device("cuda")

# First reset the op profiler cache.
ResetCache(device)

res = Profile(expr, device)
lat, ws_size = res["latency"], res["workspace_size"]
print("Latency:", lat[0].value) # By default it only profiles once.
print("Workspace size:", ws_size.value)
```

Output:

```
Latency: 1764.460693359375
Workspace size: 40566096.0
```

The units of latency and workspace size are us and bytes, respectively. Note that workspace size is determined by the kernel implementation. In the above example, Conv2D is offloaded to NVIDIA CuDNN, and some CuDNN Conv2D algorithms require additional workspace space memory during the computation, so the workspace size is larger than 0. However, CuDNN may not select the same algorithm, so you may see different workspace sizes if you run the above example locally.

## Profile Memory

To profile the memory footprint over the execution, we can simply wrap the model execution with the RAF profiler APIs:

```python
# Setup model.
optimizer, args = setup_model()

# Clean the existing profiled stats and enable memory profiler.
raf.utils.memory_profiler.reset()
raf.utils.memory_profiler.start()

# Run the model. Note that we do not need to warm up before profiling memory.
run_vm_model(optimizer, "cuda", args)

# Disable memory profiler.
raf.utils.memory_profiler.stop()

# Show the peak memory usage.
result = raf.utils.memory_profiler.get_max_memory_info(raf.Device("cuda"))
print("Max used memory (MBs):", result["max_used"].value)
print("Max allocated memory (MBs):", result["max_allocated"].value)

# Show the memory traces.
print("Memory Trace:")
print(raf.utils.memory_profiler.get_memory_trace(raf.Device("cuda")))
```

We could usually get some insights from the memory profiling results:

1. If maximum used memory is much more smaller than the maximum allocated memory, it might indidate that the memory fragmentation is serious in your model execution.
2. Look into the memory trace, you can find that the peak memory usually happens at the point of calculating the loss, because all required intermediate tensors are already generated at this point.
3. If you want to reduce the memory footprint, it is usually a good idea to find the point that has a big bump of the memory consumption, and see if you could reduce the tensor shape or dependency.
