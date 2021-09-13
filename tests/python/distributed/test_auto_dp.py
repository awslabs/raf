# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest

import mnm
from mnm.model import Conv2d, Linear, BatchNorm
from mnm import distributed as dist
from mnm.testing import randn, one_hot_torch, run_vm_model

import tvm


class MNMTest(mnm.Model):
    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape=28, num_classes=10):
        self.conv1 = Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            bias=False)
        self.bn1 = BatchNorm(6)
        self.linear1 = Linear((input_shape // 2) ** 2 * 6,
                              num_classes)
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


# pylint: disable=unused-variable
@pytest.mark.skip()
def test_dp():
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    device = f"cuda({dctx.local_rank})"
    tvm_device = tvm.nd.device("cuda")

    m_model = MNMTest()
    m_model.to(device=device)
    m_param_dict = m_model.state()
    m_model.train_mode()

    m_x, _ = randn([1, 3, 28, 28], device=device, requires_grad=True)
    m_y, _ = one_hot_torch(batch_size=1, num_classes=10, device=device)
    m_dy, _ = randn((), device=device)

    model_train = mnm.optim.sgd.with_sgd()(m_model)
    run_vm_model(model_train, device, [m_dy, m_x, m_y])
    tvm_device.sync()
    dctx.enable_data_parallel = False
    dist.RemoveCommunicator()


if __name__ == "__main__":
    test_dp()
