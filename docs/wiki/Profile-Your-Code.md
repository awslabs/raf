
A simple profiler for meta.

# How to use

## Start Profiling

call `mnm.utils.profiler.start()` when you want to profile.
The execution before this calling will not be profiled.

## Stop Profiling

call `mnm.utils.profiler.stop()` when you want to stop profiling.
The execution after this calling will not be profiled.

## Get Results

call `mnm.utils.profiler.get()` when you want to collect all the profiling results. The profiling results will be stored in memory in json format. If you want to store the results into disk dirtectly without storing in memory first, you can call `mnm.utils.profiler.dump(filename:str)`.

## Visualize Profling

Open the file that stores the profiling results with chrome:://tracing.
(Tips: open your chrome, and type "chrome::tracing", then drag your file into it.)

# Functionality

We can profile both execution on cpu and asynchronous execution on gpu.

## Execution on cpu

We profiled the interpreting of the PrimitiveOpEnv, including resources request/free, op execution. (See impl/Interpreter.cc for more info)

## Execution on gpu

We profiled the execution of `mnm.op.matmul*` using cudaEvent.

# More Profiling Contents

If you want profile more content in the backend, you can add your own profiling code following the followed instructions.

## Add profile of cpu execution

### Step1: include

Add `#include "mnm/profiler.h"` before using it.

### Step2: Wrap the code snippet with Profiler macros

We provide a macro to profile your code conviently. To use it, you just need to wrap your code into this marco.
Here is an example:

``` cpp
// The code snippet you want to profile
for (int i = 0, n = req->memory.size(); i < n; ++i) {
  RequestMemory(req.get(), i);
}

```

``` cpp
// Using BASE_PROFLIER to profile, the code snippet is the last argument of this macro.
WITH_BASE_PROFILER(call->ctx, op->name, "MemoryRequest",
              {"Count: " + std::to_string(req->memory.size())}, {
                for (int i = 0, n = req->memory.size(); i < n; ++i) {
                  RequestMemory(req.get(), i);
                }
              });
```

### Example

`mnm/src/impl/interpreter.cc`

## Add profile of gpu execution

we use `ProfilerCudaHelper` instead of `ProfileHelper`.

### Step1 include

Add `#include "mnm/src/profiler/cuda/cuda_profiler.h"` before using it.

### Step2 Wrap the code snippet with Cuda Profiler macros

Just like the base profiler, we also provide a macro WITH_CUDA_PROFILER for profiling on cuda asynchronous execution.
The usage is same to WITH_BASE_PROFILER.

### Example

`mnm/src/op/dispatch/cublas/matmul.cc`
