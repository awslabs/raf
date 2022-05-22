<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# How to Train Your Model

This article introduces how to build and train a model in RAF. Note that if you write a model in PyTorch and would like to train in with RAF, please refer to [Train PyTorch Modele](./Train-PyTorch-Model.md).

The programming model of implementing a deep learning mode in RAF is basically the same as PyTorch. We use ResNet as an example:

```python
import raf
from raf.model import Conv2d, BatchNorm, Sequential, Linear
from raf.testing import randn_torch, get_vm_executor, run_vm_executor, one_hot_torch

class RAFBottleneck(raf.Model):
    expansion = 4

    def build(self, inplanes, planes, stride):
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
            groups=1,
            dilation=1,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        if stride != 1 or inplanes != planes * RAFBottleneck.expansion:
            self.downsample = Sequential(
                Conv2d(
                    inplanes,
                    planes * RAFBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * RAFBottleneck.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = raf.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = raf.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = raf.add(out, identity)
        out = raf.relu(out)
        return out

class RAFResNet50(raf.Model):
    def build(self, num_blocks, num_classes=1000):
        self.num_blocks = num_blocks
        self.inplanes = 64
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(self.inplanes)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.fc1 = Linear(512 * RAFBottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [RAFBottleneck(self.inplanes, planes, stride)]
        self.inplanes = planes * RAFBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(RAFBottleneck(self.inplanes, planes, stride=1))
        return Sequential(*layers)

    @raf.model.trace
    def forward_infer(self, x):
        """Forward function for inference that outputs the inference result."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = raf.relu(x)
        x = raf.max_pool2d(x, kernel=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = raf.avg_pool2d(x, kernel=7, stride=7)
        x = raf.batch_flatten(x)
        x = self.fc1(x)
        return x

    @raf.model.trace
    def forward(self, x, y_true):
        """Forward function for training that outputs the loss."""
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss
```

Then wrap the optimizer for training:

```python
device = "cuda"
model = RAFResNet50([3, 4, 6, 3])
model.to(device=device)
model.train_mode()

# Wrap the SGD optimizer.
optimizer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(model)
```

Until this point, our model is ready for training.

## Train model with virtual machine (recommended)

RAF offers a virtual machine (VM) runtime to execute the model training process. Compare to the interpreter that executes the model graph op-by-op, VM first applies many graph-level optimizations, such as constant folding, expression simplification, fusion, memory planning, and so on. As a result, it is highly recommended to use VM to train your model. Here is an example training script that trains the model with RAF VM. Note that we randomly generate the dataset for training in this example.

```python
batch_size = 8
input_shape = (batch_size, 3, 224, 224)
dy, _ = randn_torch((), std=0.0, mean=1.0, requires_grad=False, device=device)  # dy = tensor(1.0)

# Training loop
num_epoch = 2
record = None
executor = None
for _ in range(num_epoch):
    # Prepare input data, use random data as example here
    r_x, _ = randn_torch(input_shape, device=device, dtype="float32")
    r_ytrue, _ = one_hot_torch(size=batch_size, num_classes=10, device=device)
    args = [dy, r_x, r_ytrue]

    # Initialize the VM at the first iteration.
    if record is None and executor is None:
        record = optimizer._internal(*args)
        executor = get_vm_executor(record.mod, device)

    ret = run_vm_executor(executor, record, args, device)
    loss = ret[0][0]  # ret[0] for some models
    print("Loss:", loss.numpy())
```

One major different as PyTorch is RAF needs to initialize a virtual machine in the first iteration. The initialization involves graph level optimization and VM bytecode compilation. In addition, when running the VM executor in the first iteration, RAF performs just-in-time (JIT) compilation to code generate each kernel, so it may take a bit longer.

## Train model with interpreter (testing prupose)

If you just want to test the model output but do not care about the performance, interpreter serves for this purpose. To use the interpreter, you could simply run the model as follows:

```python
batch_size = 8

dy, _ = randn_torch((), std=0.0, mean=1.0, requires_grad=False, device=device)  # dy = tensor(1.0)
r_x, _ = randn_torch(input_shape, device=device, dtype="float32")
r_ytrue, _ = one_hot_torch(size=batch_size, num_classes=10, device=device)
args = [dy, r_x, r_ytrue]

ret = optimizer(*args)
loss = ret[0]
print("Loss:", loss)
```
