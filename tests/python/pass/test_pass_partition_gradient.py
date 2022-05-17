# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-arguments, protected-access, attribute-defined-outside-init
# pylint: disable=too-many-locals, too-many-branches
import re
from unittest.mock import patch
import pytest

import raf
from raf._ffi.pass_ import PartitionGradient, InferType
from raf.frontend.model import FrameworkModel
from raf.model import BatchNorm, Conv2d, Linear
from raf.optim.optim import with_autodiff
from raf.testing import compile_vm_model, randn, one_hot_torch


def verify_ir(opt_level, ad_model, args, rank_size, rank, n_grad, n_pad):
    bucket_size = 5000000000
    record = ad_model._internal(*args)
    mod = record.mod
    mod = InferType()(mod)
    mod = PartitionGradient(opt_level, rank_size, rank, bucket_size)(mod)
    mod = InferType()(mod)
    text = raf.ir.AsText(mod)
    assert text.count("raf.op.pad") == n_pad
    if opt_level == 1:
        assert text.count("raf.op.split") == n_grad

    nccl_version = raf.build.with_nccl()

    if opt_level == 1:
        # Gradients will be sliced as follows in ZeRO-1. Assuming rank_size=4 and rank=3:
        # let %x_1 = raf.op.split(%a0, 4);
        # let %x_2 = %x_1.3;
        # ...
        # let %a10 = (..., %x_2, ...);
        grad_regex = rf"let %x_(\d+) = %x_\d+\.{rank};"
    elif opt_level == 2:
        # ZeRO-2 uses group_reduce_scatter to slice gradients.
        assert text.count("raf.op._group_reduce_scatter") == 1, text

        # Gradients will be sliced as follows in ZeRO-2:
        # # if NCCL version is >= 2.10
        # let %x = (%x_0, %x1, ...);
        # let %y = raf.op._group_recuce_scatter(%x, avg);
        # let %y_0 = %y.0
        # let %y_1 = %y.1
        # ...
        # let %a10 = (..., %y_0, %y_1, ...);
        # # else NCCL version is < 2.10
        # let %x = (%x_0, %x1, ...);
        # let %y = raf.op._group_recuce_scatter(%x, avg);
        # let %y_0 = %y.0
        # let %y_3 = raf.op.divide(%y_0, ...)
        # ...
        # let %a10 = (..., %y_3, ...);
        if nccl_version >= 21000:
            grad_regex = rf"let %x_(\d+) = raf.op._group_reduce_scatter.+"
        else:
            grad_regex = rf"let %x_(\d+) = raf.op.divide.+"
    else:
        assert False, "Unsupported opt_level %d" % opt_level

    # Verify that the output gradient tuple contains all sliced gradients.
    verify_grad_tuple = False
    split_grads = set()
    find_grad = False
    for line in text.split("\n"):
        tokens = re.search(grad_regex, line)
        if tokens:
            grad_var = f"%x_{tokens.group(1)}"
            split_grads.add(grad_var)
            find_grad = True
            continue
        if find_grad:
            if opt_level == 2 and nccl_version >= 2100:
                pattern = rf"let %x_(\d+) = {grad_var}.+;"
                tokens = re.search(pattern, line)
                if tokens:
                    if grad_var in split_grads:
                        split_grads = set()
                    split_grads.add(f"%x_{tokens.group(1)}")
                    continue

            tokens = re.search(r"let .+ = \((.+)\);", line)
            if tokens:
                if all([g in split_grads for g in tokens.group(1).replace(" ", "").split(",")]):
                    verify_grad_tuple = True
                    break
    assert verify_grad_tuple

    if raf.build.with_cuda():
        model = FrameworkModel(mod, mod, ad_model.state(), {})
        compile_vm_model(model, "cuda", [arg.to(device="cuda") for arg in args])


@patch("raf.distributed.get_communicator")
@patch("raf.distributed.get_config")
@pytest.mark.parametrize("opt_level", [1, 2])
@pytest.mark.parametrize("batch", [7, 8])
def test_basic(mock_get_config, mock_get_comm, opt_level, batch):
    class Model(raf.Model):
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

    # Mock dist config & communicator to let with_lans generate the desired IR.
    class MockConfig:
        def __init__(self):
            self.enable_data_parallel = True
            self.zero_opt_level = opt_level

    mock_get_config.return_value = MockConfig()

    class MockComm:
        def __init__(self):
            self.size = 4
            self.rank = 3

    mock_get_comm.return_value = MockComm()
    if opt_level == 2 and raf.build.with_nccl() is None:
        pytest.skip("NCCL is not supported")

    model = Model()
    ad_model = with_autodiff(model)
    m_x, _ = randn((batch, 3, 28, 28), dtype="float32")
    m_dy, _ = randn((), dtype="float32")
    m_ytrue, _ = one_hot_torch(size=batch, num_classes=10)

    if batch == 8:
        # The gradient of conv2d_dx and nll_loss is dividable so no padding is need.
        verify_ir(opt_level, ad_model, [m_dy, m_x, m_ytrue], 4, 0, 9, 7)
    elif batch == 7:
        # The first axis of all gradients are non-dividable.
        verify_ir(opt_level, ad_model, [m_dy, m_x, m_ytrue], 4, 1, 9, 9)


if __name__ == "__main__":
    pytest.main([__file__])
