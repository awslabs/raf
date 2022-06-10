<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# How to Train PyTorch Models

In this article, we demonstrate how to train a PyTorch model in RAF.

Before start, please install the following dependencies at first:

- PyTorch
- Torchvision (optional)
- Hugging Face Transformers (optinal)

You can install them via pip, or refer to: https://pytorch.org/get-started/locally/ and https://huggingface.co/docs/transformers/installation

```
pip install torch torchvision transformers
```

The process to train a PyTorch model in RAF consists of importing model and writing training script, the following up sections will introduce them.

## Import PyTorch Model

Users can use RAF frontend API to obtain RAF model from PyTorch model. In addition, RAF provides short-cut APIs to help users import model from torchvision or transformers.

An example of import torch model directly:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import raf
from raf.frontend import from_pytorch
from raf.testing import randn_torch, one_hot_torch, get_vm_executor, run_vm_executor


class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.sigmoid(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

shape_dict = {"input0": ((32, 3, 28, 28), "float32")}
device = "cuda"
input_shape, dtype = list(shape_dict.values())[0]
batch_size = input_shape[0]

t_model = TorchLeNet()
r_model = from_pytorch(t_model, shape_dict)
r_model.to(device=device)
r_model.train_mode()
```

Now we get the imported model. Note that there is no loss function in this model, you can append loss function to the converted model or write it in the torch model definition directly.

```python
from raf._op import sym

# prepare random data, they just provides shape and dtype info
r_x, _ = randn_torch(input_shape, device=device, dtype=dtype)
r_ytrue, _ = one_hot_torch(size=batch_size, num_classes=10, device=device)

out = r_model.record(r_x)
y_pred = sym.log_softmax(out)
loss = sym.nll_loss(r_ytrue, y_pred)
r_model_w_loss = r_model + loss
```

If you want to use models in transformers or torchvision:

```python
from raf.testing import get_transformer_model

model, out_shape = get_transformer_model("bert-base-uncased", batch_size=32, seq_length=128, dtype="float32")
model.to(device=device)
model.train_mode()
```

```python
from raf.testing import get_torchvision_model

model, out_shape = get_torchvision_model("resnet18", batch_size=32, image_size=(224, 224), dtype="float32")
model.to(device=device)
model.train_mode()
```

## Write Training Script

In order to train a model, in addition to loss function, we also need an optimizer. For a RAF model already with loss function, just use RAF optimizer API to wrap the model:

```python
r_trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(r_model_w_loss)
```

Actually, we can append loss function and optimizer in one line:

```python
from raf.testing import append_loss_n_optimizer

r_trainer = append_loss_n_optimizer(r_model, args=[r_x], out_shape=(32, 10), y_true=r_ytrue, optimizer="sgd")
```

Finally we can write the training script and train the model:

```python
dy, _ = randn_torch((), std=0.0, mean=1.0, requires_grad=False, device=device)  # dy = tensor(1.0)

# get vm exec
record = r_trainer._internal(dy, r_x, r_ytrue)
executor = get_vm_executor(record.mod, device)

# training loop
for _ in range(num_epoch):
    # prepare input data, use random data as example here
    r_x, _ = randn_torch(input_shape, device=device, dtype=dtype)
    r_ytrue, _ = one_hot_torch(size=batch_size, num_classes=10, device=device)
    args = [dy, r_x, r_ytrue]
    ret = run_vm_executor(executor, record, args, device)
    loss = ret[0]  # ret[0][0] for some models
    print("LOSS:", loss)
```
