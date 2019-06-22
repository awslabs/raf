import mnm
import torch
import torch.nn.functional as F

from mnm import cpu
from mnm.value import TensorValue

from utils import invoke_make_output


def test_conv2d():
    n, c, h, w = 5, 3, 128, 64
    c_out, c_in, kh, kw = 10, c, 5, 7
    stride = (2, 3)
    padding = (1, 10)
    dilation = (3, 6)
    groups = 1
    data = torch.empty(n, c, h, w)
    weight = torch.empty(c_out, c_in, kh, kw)
    out = F.conv2d(data, weight, None, stride, padding, dilation, groups)
    torch_shape = tuple(out.shape)
    data = TensorValue.assemble((n, c, h, w), "float32", cpu())
    weight = TensorValue.assemble((c_out, c_in, kh, kw), "float32", cpu())
    out = invoke_make_output("mnm.op.conv2d", "Conv2DAttrs",
                             (data, weight), stride, padding, dilation, groups)
    mnm_shape = out.shape
    assert mnm_shape == torch_shape


if __name__ == "__main__":
    test_conv2d()
