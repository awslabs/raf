# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-variable,protected-access
from unittest.mock import patch
import pytest

import raf
from raf.model import Conv2d, Linear, BatchNorm
from raf.testing import one_hot_torch, randn
from raf._ffi import pass_


class RAFTest(raf.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

    # pylint: enable=attribute-defined-outside-init

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


def optimize(mod):
    mod = pass_.ToGraphNormalForm()(mod)
    mod = pass_.ToBasicBlockNormalForm()(mod)
    mod = pass_.ToANormalForm()(mod)
    mod = pass_.InferType()(mod)
    mod = pass_.GroupAllgather()(mod)
    return mod


def lower(model, args):
    mod = model._internal(*args).mod
    return optimize(mod)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@patch("raf.distributed.get_communicator")
@patch("raf.distributed.get_config")
def test_group(mock_get_config, mock_get_comm):
    # pylint: disable=too-many-locals, protected-access
    # Mock the context to let with_lans generate the desired IR.
    class MockConfig:
        def __init__(self):
            self.enable_data_parallel = True
            self.zero_opt_level = 2
            self.group_bucket_size = 5000000000

    mock_get_config.return_value = MockConfig()

    class MockComm:
        def __init__(self):
            self.size = 4
            self.local_rank = 0
            self.rank = 3

    mock_get_comm.return_value = MockComm()
    shape, n_classes = 28, 10
    batch_size = 7
    m_model = RAFTest(shape, 10)
    m_model.train_mode()
    m_optimizer = raf.optim.lans.with_lans()(m_model)

    device = "cuda"
    m_x, _ = randn([batch_size, 3, shape, shape], requires_grad=True, device=device)
    m_dy, _ = randn((), device=device, requires_grad=False)
    m_ytrue, _ = one_hot_torch(size=batch_size, num_classes=n_classes, device=device)
    args = [m_dy, m_x, m_ytrue]

    record = m_optimizer._internal(*args)
    mod = record.mod

    func = lower(m_optimizer, [*args])["main"]
    text = raf.ir.AsText(func)
    ## Verify IR. This model has 7 parameters and 9 gradients
    ## There should be only 1 group GroupAllgather that perform
    ## group operations for 9 gradients.
    assert text.count("raf.op._group_allgather") == 1, text
    assert text.count("raf.op.strided_slice") == 7, text
    ## Using "zeros(" to exclude "zeros_like" op.
    assert text.count("raf.op.zeros(") == 7, text


if __name__ == "__main__":
    pytest.main([__file__])
