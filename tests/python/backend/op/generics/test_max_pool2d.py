import torch
import torch.nn.functional as F

from mnm._core.op import invoke_make_output
from mnm._core.value import TensorValue


def test_max_pool2d():
    n, c, h, w = 5, 3, 128, 64
    kh, kw = 5, 7
    stride = (2, 3)
    padding = (1, 0)
    dilation = (3, 6)
    ceil_mode = False
    data = torch.empty(n, c, h, w)
    out = F.max_pool2d(data, (kh, kw), stride, padding,
                       dilation, False, ceil_mode)
    torch_shape = tuple(out.shape)
    data = TensorValue.assemble((n, c, h, w), "float32", "cpu")
    out = invoke_make_output("mnm.op.max_pool2d", "mnm.attrs.MaxPoolAttrs", data,
                             kernel=(kh, kw),
                             stride=stride, padding=padding,
                             dilation=dilation, ceil_mode=ceil_mode)
    mnm_shape = out.shape
    assert(mnm_shape == torch_shape)


if __name__ == "__main__":
    test_max_pool2d()
