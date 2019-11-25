import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mnm


@pytest.mark.skipif(mnm._ffi.build_info.use_cuda() == "OFF", reason="CUDA is not enabled")  # pylint: disable=protected-access
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1, 2, 4])
def test_mnm_conv2d(stride, dilation, padding):
    # N.B.: NCHW + OIHW
    x = np.random.randn(8, 3, 32, 32).astype('float32')
    w = np.random.randn(16, 3, 3, 3).astype('float32')
    t_x = torch.Tensor(x)
    t_w = torch.Tensor(w)
    x = mnm.array(x, ctx='cuda')
    w = mnm.array(w, ctx='cuda')
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    y = mnm.conv2d(x, w, stride=stride, dilation=dilation, padding=padding)
    np.testing.assert_allclose(y.asnumpy(), t_y.numpy(), rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    pytest.main()
