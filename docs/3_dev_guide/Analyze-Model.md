<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Analyze Your Model

To optimize the model training process, you may want to know more details about your model in addition to its network architecture. In this short article, we introduce RAF built-in tools that could help you understand the model from system perspective. Specifically, RAF provides a few useful APIs to analyze model parameter sizes, computation GFLOPS, and memory footprints.

## Analyze Parameters

Parameters are defined as the model allocated tensors that could be learned during the training process. For example, the weights of linear and convolutional layers in CNNs are parameters because they are float types and learnable. On the other hand, the position IDs of embedding layers in transformer models are NOT parameters because they are integers and cannot be learned.

In RAF, you can easily obtain the parameter size of your model to judge the batch size and worker number for training.

```python
import raf
from raf.model.model import get_param_size
from raf.testing import get_transformer_model

# Load a transformer model. Note that transformers package has to be available.
model, _ = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")

# Switch to training model; otherwise no weights are learnable.
model.train_mode() 

print("Parameter #:", get_param_size(model))
print("Parameter size (MBs):", get_param_size(model, mbs=True))
```

Output:

```
Parameter #: 109514298.0
Parameter size (MBs): 417.7638931274414
```

The above example loads a BERT base model from Huggingface and analyzes its parameters. We can see that it has 109M parameters, and their total sizes are 417MBs (109514298 * 4 bytes / 1048576) because we use float32 data type. It means you could consider using float16 to reduce the parameter size by a half if this matters to you.

## Analyze Memory Footprint

In addition to the parameters, other important sources of consuming memory include intermediate tensors and worksapces. Intermediate tensors cannot be released until their last use. For example, the forward activations are intermediate tensors we need to keep for backward propagation. Workspaces are the memory buffers requested by some operator kernels as temporary buffers during the computation. Although workspaces can always be released right after each kernel execution, it might be the last straw to out-of-memory.

Memory footprint is the series of memory consumption over model execution, including forward inference and backward propagation. Investigating memory footprint is usually helpful if you frequently encounter out-of-memory errors during the training process, because memory profiler works only when you could finish running the model.

In RAF, we provide an API to analyze the memory footprint of a model even it may use more memory than the device capacity. Specifically, it traverses each op in the VM execution order and calculates the memory consumption at each moment of invoking an operator kernel. Note that it may take a while to generate the memory footprint because the analyzer performs JIT to include workspace memory usages.

```python
import raf
from raf.model.model import trace_memory, get_peak_memory
from raf.testing import get_transformer_model, randint, randn_torch

# Load a transformer model. Note that transformers package has to be available.
model, _ = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")

# Generate a dummy input.
r_x, _ = randint((32, 128), low=0, high=10000, dtype="int64")

# Analyze memory footprint.
trace = trace_memory(model, "cuda", [r_x])
print("Trace (MBs):")
for name, mem in trace:
    print(name, mem)

# Analyze peak memory.
print("Peak (MBs):", get_peak_memory(model, "cuda", [r_x]))
```

Output:

```
Estimating memory footprint...
Estimated 0.00% (0/1833)
Estimated 9.98% (183/1833)
Estimated 19.97% (366/1833)
Estimated 29.95% (549/1833)
Estimated 39.93% (732/1833)
Estimated 49.92% (915/1833)
Estimated 59.90% (1098/1833)
Estimated 69.89% (1281/1833)
Estimated 79.87% (1464/1833)
Estimated 89.85% (1647/1833)
Estimated 99.84% (1830/1833)
Trace (MBs):
fused_embedding_cast_embedding_add_strided_slice_embedding_add 429.7990417480469
raf_op_tvm_layer_norm 441.7990417480469
...
Estimating memory footprint...
Peak (MBs): 1371.611572265625
```

As can be seen from the above example, BERT-base requires 1371 MBs memory when batch size is 32 and sequence length is 128. On the other hand, since the above model only includes the forward graph, its peak memory would be much lower than the one with backward graph. To experiment, we apply the SGD optimizer and repeat the process:

```python
import raf
from raf.model.model import get_peak_memory
from raf.testing import get_transformer_model, append_loss_n_optimizer
from raf.testing import randn_torch, one_hot_torch

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

# Append loss and optimizer, which also includes AutoDiff.
optimizer = append_loss_n_optimizer(model, [r_x], shape, r_ytrue)

# Analyze peak memory.
print("Peak (MBs):", get_peak_memory(optimizer, "cuda", [r_dy, r_x, r_ytrue]))
```

Output:

```
Estimating memory footprint...
Estimated 0.00% (0/8066)
Estimated 9.99% (806/8066)
Estimated 19.99% (1612/8066)
Estimated 29.98% (2418/8066)
Estimated 39.97% (3224/8066)
Estimated 49.96% (4030/8066)
Estimated 59.96% (4836/8066)
Estimated 69.95% (5642/8066)
Estimated 79.94% (6448/8066)
Estimated 89.93% (7254/8066)
Estimated 99.93% (8060/8066)
Peak (MBs): 5640.39990234375
```

There are two differences compared with the previous one. First, the total execution kernels is increased from 1833 to 8066, meaning that the backward propagation and optimizer logic has been added. Second, the peak memory is increased from 1372 MBs to 5640 MBs, which includes optimizer state as well as some forward intermediate results required by backward for gradient computations.

## Analyze Computation GFLOPS

Finally, you may be also interested in how complex your model execution is. A useful metric to get a sense is the computation GFLOPS, which is the total compute operators (e.g., multiply and addition) required by your model. In RAF, we provide the following API to analyze the computation GFLOPS of the given model.

```python
import raf
from raf.model.model import calc_model_gflops
from raf.testing import get_transformer_model, randint

# Load a transformer model. Note that transformers package has to be available.
model, _ = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")

# Generate dummy inputs.
r_x, _ = randint((32, 128), low=0, high=10000, dtype="int64")

print("GFLOPS:", calc_model_gflops(model, "cuda", [r_x]))
```

Output:

```
GFLOPS: 914.1637796584982
```
