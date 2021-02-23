# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import numpy as np
import mnm
from mnm.model import Conv2d, Linear, BatchNorm
from mnm import distributed as dist


def one_hot(batch_size, num_classes, ctx="cuda", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    assert list(m_x.shape) == [batch_size, num_classes]
    return m_x


def randn(shape, *, ctx="cuda", dtype="float32", std=1.0, mean=0.0,
          requires_grad=False, positive=False):
    if positive:
        x = np.abs(np.random.randn(*shape)) * std + mean
    else:
        x = np.random.randn(*shape) * std + mean
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    if requires_grad:
        m_x.requires_grad = True
    return m_x


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
def run(config):
    dctx = dist.get_context()
    dctx.enable_data_parallel = True
    ctx = f"cuda({dctx.local_rank})"

    batch_size = config.get("batch_size", 1)
    input_shape = config.get("input_shape", 28)
    num_classes = config.get("num_classes", 10)
    num_channels = 3

    m_model = MNMTest(input_shape, num_classes)
    m_model.to(ctx=ctx)
    m_model.train_mode()

    shape = [batch_size, num_channels, input_shape, input_shape]
    m_x = randn(shape, ctx=ctx, requires_grad=True)
    m_y = one_hot(batch_size=batch_size, num_classes=num_classes, ctx=ctx)

    loss = m_model(m_x, m_y)
    print("The loss of single iteration: ", loss)

    dctx.enable_data_parallel = False


if __name__ == "__main__":
    if mnm.build.with_cuda():
        config = {
            "batch_size": 1,
            "input_shape": 28,
            "num_classes": 10
        }
        run(config)
    else:
        print("You must enable Cuda for distributed training.")
