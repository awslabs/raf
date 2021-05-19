# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-member,no-self-use
import pytest
import torch
import torch.nn.functional as F

import mnm
from mnm.testing import randn_torch, run_vm_model, check, with_backend


@pytest.mark.skipif(not mnm.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize("shapes", [
    ((4, 256, 32, 32), (64, 256, 1, 1)),
    ((8, 3, 32, 32), (16, 3, 3, 3)),
])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_conv2d_relu(shapes, stride, dilation, padding):
    device, dtype = "cuda", "float32"
    class Conv2D(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, w):
            y = mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1,
                           layout="NHWC", kernel_layout="OHWI", out_layout="NHWC")
            y = mnm.relu(y)
            return y

    model = Conv2D()
    xshape, wshape = shapes
    m_x, t_x = randn_torch(xshape, device=device, dtype=dtype)
    m_w, t_w = randn_torch(wshape, device=device, dtype=dtype)
    m_x = mnm.transpose(m_x, (0, 2, 3, 1))
    m_w = mnm.transpose(m_w, (0, 2, 3, 1))
    m_y = with_backend("cutlass")(run_vm_model)(model, device, [m_x, m_w])
    t_model = torch.nn.Conv2d(
        wshape[3], wshape[0], (wshape[1], wshape[2]),
        stride=stride, padding=padding, dilation=dilation, bias=False).cuda().float()
    t_model.weight = torch.nn.Parameter(t_w)
    t_model = t_model.to(memory_format=torch.channels_last)
    t_y = t_model(t_x)
    t_y = F.relu(t_y.permute(0, 2, 3, 1))
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
