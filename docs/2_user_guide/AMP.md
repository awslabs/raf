<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Automatic Mixed Precision (AMP)

In this article, we introduce how to enable AMP in model training for better performance. In short, the RAF AutoCast pass will transform the model IR and insert some `cast` ops for two purposes. First, compute-intensive ops can be executed with AMP data type (float16 or bfloat16) to reduce the latency. Second, some tensors can be stored in AMP data data type to reduce the memory footprint. For the general introduction to AMP, please refer to [this doc](https://developer.nvidia.com/automatic-mixed-precision).

Here is an example of enabling AMP in RAF:

```python
import time
import transformers

import raf
from raf.model.model import get_peak_memory
from raf.testing import get_transformer_model_by_config, append_loss_n_optimizer
from raf.testing import randn_torch, one_hot_torch, randint
from raf.testing import profile_vm_model

def setup_model():
  # We use one layer BERT in this example.
  config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
  config.num_hidden_layers = 1

  # Load a transformer model.
  model, shape = get_transformer_model_by_config(
    config, batch_size=8, seq_length=128, dtype="float32"
  )

  # Generate dummy inputs.
  r_x, _ = randint((8, 128), low=0, high=10000, dtype="int64")
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

# Profile the model. The "profile_vm_model" API runs the model multiple times and reports the per-iteration latency.
device = "cuda"
print("FP32 (ms):", profile_vm_model(optimizer, device, args, opt_level=3, warmup=5, number=10, repeat=1)[0])
print("FP32 (MBs):", get_peak_memory(optimizer, device, args))

# AMP model.
amp_optimizer = raf.amp.autocast(optimizer, args)
print("AMP (ms):", profile_vm_model(amp_optimizer, device, args, opt_level=3, warmup=5, number=10, repeat=1)[0])
print("AMP (MBs):", get_peak_memory(amp_optimizer, device, args))
```

Output:

```
FP32 (ms): 60.54600143432617
FP32 (MBs): 751.02685546875
AMP (ms): 28.728599548339844
AMP (MBs): 701.22900390625
```

We can see that by applying the `raf.amp.autocast`, the per-iteration latency is reduced by almost a half, because compute-intenstive ops are now executed with float16 data type and result in about a half latency. Meanwhile, since the parameter size of this BERT model is 120.34 MBs, the peak intermetidate memory is also reduced from 751-120=631 to 701-120=581 MBs.

## Casting Rules and Customization

As we mentioned in the beginning of this article, the core idea of AMP models compared to pure float16 models is to keep some arithmetic sensitive ops, such as `softmax`, running with float32. As a result, we need to define how an op should be casted so that the AutoCast pass could insert `cast` ops accordingly. If you are interested in casting rules, they are located in `python/raf/amp/type_hints.py`. Here is an example cast rule for `bias_add`. Since it can be executed with either float32 or float16, the casting rule is just following the input data type to minimize the casting overhead.

```python
def cast_bias_add(args, ret_type, amp_dtype):
    # Follow the input data type
    target_dtype = amp_dtype if check_dtype(args[0].checked_type, amp_dtype) else "float32"

    # Generate type hint for each input argument.
    return [gen_hint_helper(args[0].checked_type, target_dtype), # input
            gen_hint_helper(args[0].checked_type, target_dtype), # bias
            PrimType(None) # axis, not floating data type so don't touch
           ]
```

On the other hand, if you want to override existing cast rules for testing or any purpose, here is an example:

```python
def never_cast_bias_add(args, ret_type, amp_dtype):
    """Custom bias add casting rule."""
    return [PrimType("float32"), PrimType("float32"), PrimType(None)]

with raf.amp.CustomTypeHint({"raf.op.bias_add": never_cast_bias_add}):
    amp_model = raf.amp.autocast(model)
    # ...
```
