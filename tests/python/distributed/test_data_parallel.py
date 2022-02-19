# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=attribute-defined-outside-init,protected-access,too-many-locals
# pylint: disable=too-many-statements,invalid-name
import sys
import pytest

import mnm
from mnm.model import Conv2d, Linear, BatchNorm
from mnm import distributed as dist
from mnm.testing import randn, one_hot_torch, run_vm_model, skip_dist_test, with_seed

import tvm


class MNMTest(mnm.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

    # pylint: enable=attribute-defined-outside-init

    @mnm.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        y_pred = mnm.log_softmax(y_pred)
        loss = mnm.nll_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @mnm.model.trace
    def forward_infer(self, x):
        out = self.bn1(self.conv1(x))
        out = mnm.sigmoid(out)
        out = mnm.avg_pool2d(out, (2, 2), (2, 2))
        out = mnm.batch_flatten(out)
        out = self.linear1(out)
        return out


def run_model(device):
    tvm_device = tvm.nd.device("cuda")

    m_model = MNMTest()
    m_model.to(device=device)
    m_model.train_mode()

    m_x, _ = randn([4, 3, 28, 28], device=device, requires_grad=True)
    m_y, _ = one_hot_torch(batch_size=4, num_classes=10, device=device)
    m_dy, _ = randn((), device=device)

    model_train = mnm.optim.sgd.with_sgd()(m_model)
    run_vm_model(model_train, device, [m_dy, m_x, m_y])
    tvm_device.sync()


@pytest.mark.skipif(
    skip_dist_test(min_rank_num=2),
    reason="Distribution is not enabled or only one device is available",
)
@with_seed(0)
def test_data_parallel():
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    device = f"cuda({dctx.local_rank})"

    run_model(device)

    dctx.enable_data_parallel = False


@pytest.mark.skipif(
    skip_dist_test(min_rank_num=2),
    reason="Distribution is not enabled or only one device is available",
)
@with_seed(0)
def test_zero_opt_1():
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    dctx.zero_opt_level = 1
    device = f"cuda({dctx.local_rank})"

    run_model(device)

    dctx.enable_data_parallel = False
    dctx.zero_opt_level = 0


if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
