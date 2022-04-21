# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,protected-access,
# pylint: disable=too-many-locals,too-many-statements
import sys

import numpy as np

import raf
import tvm
from raf.model import Conv2d, Linear, BatchNorm
from raf.model.trace import _get_func_inputs
from raf import distributed as dist
from raf import testing


class RAFTest(raf.Model):
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = raf.log_softmax(y_pred)
        loss = raf.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = raf.sigmoid(out)
        out = raf.avg_pool2d(out, (2, 2), (2, 2))
        out = raf.batch_flatten(out)
        out = self.linear1(out)
        return out


@testing.with_seed(0)
def init_model(device):
    m_model = RAFTest()
    m_model.to(device=device)
    m_model.train_mode()
    return raf.optim.sgd.with_sgd()(m_model)


@testing.with_seed(1)
def init_ref_model(device):
    m_model = RAFTest()
    m_model.to(device=device)
    m_model.infer_mode()
    return m_model


def print_at_rank_0(msg):
    comm = dist.get_communicator()
    if comm.rank == 0:
        print(msg, flush=True)


def get_vm_exec(model, device, args):
    record = model._internal(*args)
    mod = record.mod
    executor = testing.get_vm_executor(mod, device, opt_level=3)
    return record, executor


def gen_data(shape, device):
    x = np.random.uniform(low=0, high=255, size=shape)
    m_x = raf.array(x.astype("float32"), device=device)
    m_x.requires_grad = True
    return m_x


def run(train_config, raf_dist_config):
    """Train the model.

    Parameters
    ----------
    train_config: Dict[str, int]
        The training configurations.

    raf_dist_config: Dict[str, Union[bool, int]]
        The RAF distribution configurations.
    """
    # Process training configs.
    n_epoch = train_config.get("n_epoch", 1)
    n_mini_batch = train_config.get("n_mini_batch", 10)
    batch_size = train_config.get("batch_size", 16)
    input_shape = train_config.get("input_shape", 28)
    num_classes = train_config.get("num_classes", 10)
    num_channels = 3
    shape = [batch_size, num_channels, input_shape, input_shape]

    # Process distribution configs. Note that data parallel is always on.
    dcfg = dist.get_config()
    comm = dist.get_communicator()
    dcfg.enable_data_parallel = raf_dist_config.get("enable_data_parallel", False)
    dcfg.zero_opt_level = raf_dist_config.get("zero_opt_level", 0)
    device = f"cuda({comm.local_rank})"

    total_data_size = (np.prod(shape) + (batch_size * num_classes)) * n_mini_batch
    print_at_rank_0(
        "Training Config: #epoch: %d, total data set: %.2f MBs, batch: %d, shape: %s"
        % (n_epoch, (total_data_size * 4 * comm.size) / 1e6, batch_size, shape)
    )
    print_at_rank_0(
        "Distribution Config: data_parallel: %s, zero_opt_level: %d"
        % (dcfg.enable_data_parallel, dcfg.zero_opt_level)
    )

    # Fake a training data set with N mini-batches using a reference model.
    # Note that we directly initialize the data on each device. In practice,
    # we should manually distribute the training data.
    print_at_rank_0("Generating fake training data...")
    train_data = []
    ref_model_n_record = None
    for _ in range(n_mini_batch):
        m_x = gen_data(shape, device=device)
        if ref_model_n_record is None:
            ref_model_n_record = get_vm_exec(init_ref_model(device), device, [m_x])
        vm_inputs = _get_func_inputs(ref_model_n_record[0], [m_x], {}, get_handle=False)
        out = ref_model_n_record[1](*vm_inputs).numpy()
        m_y = raf.array(np.argmax(out, axis=1), device=device)
        train_data.append([m_x, m_y])

    # Initialize the model to be trained. Note that we need to fix the random seed
    # to let the model on each device has the same initial weights.
    print_at_rank_0("Initializing the model...")
    optimizer = init_model(device)

    print_at_rank_0("Start training")
    optimizer_n_record = None
    for epoch in range(n_epoch):
        losses = []
        for idx, (data, label) in enumerate(train_data):
            m_dy, _ = testing.randn((), device=device)
            args = [m_dy, data, label]

            if optimizer_n_record is None:  # JIT for the first time.
                optimizer_n_record = get_vm_exec(optimizer, device, args)

            vm_inputs = _get_func_inputs(optimizer_n_record[0], args, {}, get_handle=False)
            loss = optimizer_n_record[1](*vm_inputs)
            while isinstance(loss, (tuple, tvm.ir.container.Array, raf._core.value.TupleValue)):
                loss = loss[0]

            losses.append(testing.numpy(loss))
        print_at_rank_0("  [Epoch %2d] avg. loss: %.6f" % (epoch + 1, np.mean(losses)))

    dcfg.enable_data_parallel = False
    dcfg.zero_opt_level = 0
    dist.RemoveCommunicator()


if __name__ == "__main__":
    if not raf.build.with_distributed() or not raf.build.with_cuda():
        print("You must enable CUDA and distribution support to run this script.")
        sys.exit(0)

    train_config = {
        "n_epoch": 3,
        "n_mini_batch": 10,
        "batch_size": 16,
        "input_shape": 28,
        "num_classes": 10,
    }
    raf_dist_config = {
        "enable_data_parallel": True,
        "zero_opt_level": 1,
    }
    run(train_config, raf_dist_config)
