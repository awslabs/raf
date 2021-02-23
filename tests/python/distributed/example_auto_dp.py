# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import pytest

import mnm
from mnm.model import Conv2d, Linear, BatchNorm
from mnm import distributed as dist
from mnm import testing


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
def test_dp(config):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    dctx.overlap_comm_forward = True
    device = f"cuda({dctx.local_rank})"

    m_model = MNMTest(config[1], config[2])
    m_model.to(device=device)
    m_param_dict = m_model.state()
    m_optimizer = mnm.optim.SGD(m_param_dict.values(), 0.1, 0.01)
    m_model.train_mode()

    for i in range(config[0]):
        m_x, _ = testing.randn_torch(
            [1, 3, config[1], config[1]], device=device)
        m_x.requires_grad = True
        m_y, _ = testing.one_hot_torch(
            batch_size=1, num_classes=config[2], device=device)
        m_loss = m_model(m_x, m_y)
        m_loss.backward()
        print(f"Loss of Iter{i}: ", m_loss)
        m_optimizer.step()

    dctx.enable_data_parallel = False
    dctx.overlap_comm_forward = False
    dist.RemoveCommunicator()


if __name__ == "__main__":
    test_dp((7, 28, 10))
