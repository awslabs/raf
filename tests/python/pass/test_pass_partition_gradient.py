# pylint: disable=too-many-arguments, protected-access, attribute-defined-outside-init
import re
import pytest

import mnm
from mnm._ffi.pass_ import PartitionGradient, InferType
from mnm.frontend.model import FrameworkModel
from mnm.model import BatchNorm, Conv2d, Linear
from mnm.optim.optim import with_autodiff
from mnm.testing import compile_vm_model, randn, one_hot_torch

def verify_ir(ad_model, args, rank_size, rank, n_grad, n_pad):
    record = ad_model._internal(*args)
    mod = record.mod
    mod = InferType()(mod)
    mod = PartitionGradient(rank_size, rank)(mod)
    text = mnm.ir.AsText(mod)
    assert text.count("mnm.op.split") == n_grad
    assert text.count("mnm.op.pad") == n_pad

    # Verify that the output gradient tuple contains all sliced gradients.
    # For example, rank_size=4 and rank=3 should result in the following gradient slicing:
    # let %x_1 = mnm.op.split(%a0, 4);
    # let %x_2 = %x_1.3;
    # ...
    # let %a10 = (..., %x_2, ...);
    verify_grad_tuple = False
    split_grads = set()
    for line in text.split("\n"):
        tokens = re.search(fr"let %x_(\d+) = %x_\d+\.{rank};", line)
        if tokens:
            split_grads.add(f"%x_{tokens.group(1)}")
            continue

        tokens = re.search(r"let .+ = \((.+)\);", line)
        if tokens:
            if all([g in split_grads for g in tokens.group(1).replace(" ", "").split(",")]):
                verify_grad_tuple = True
                break
    assert verify_grad_tuple

    if mnm.build.with_cuda():
        model = FrameworkModel(mod, mod, ad_model.state(), {})
        compile_vm_model(model, "cuda", [arg.to(device="cuda") for arg in args])


@pytest.mark.parametrize("batch", [7, 8])
def test_basic(batch):
    class Model(mnm.Model):
        def build(self, input_shape=28, num_classes=10):
            self.conv1 = Conv2d(in_channels=3,
                                out_channels=6,
                                kernel_size=5,
                                padding=2,
                                bias=False)
            self.bn1 = BatchNorm(6)
            self.linear1 = Linear((input_shape // 2) ** 2 * 6, num_classes)

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

    model = Model()
    ad_model = with_autodiff(model)
    m_x, _ = randn((batch, 3, 28, 28), dtype="float32")
    m_dy, _ = randn((), dtype="float32")
    m_ytrue, _ = one_hot_torch(batch_size=batch, num_classes=10)

    if batch == 8:
        # The gradient of conv2d_dx and nll_loss is dividable so no padding is need.
        verify_ir(ad_model, [m_dy, m_x, m_ytrue], 4, 0, 9, 7)
    elif batch == 7:
        # The first axis of all gradients are non-dividable.
        verify_ir(ad_model, [m_dy, m_x, m_ytrue], 4, 1, 9, 9)

if __name__ == "__main__":
    pytest.main([__file__])
