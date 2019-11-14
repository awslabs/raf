import numpy as np
import torch
import torch.nn.functional as F

import pytest
import mnm

@pytest.mark.skipif(mnm._ffi.build_info.use_cuda() == "OFF", reason="CUDA is not enabled")  # pylint: disable=protected-access
def test_mnm_conv2d():

    x = np.random.randn(1, 3, 128, 128).astype('float32')
    w = np.random.randn(128, 3, 3, 3).astype('float32')

    t_x = torch.Tensor(x)
    t_w = torch.Tensor(w)

    x = mnm.array(x, ctx='cuda')
    w = mnm.array(w, ctx='cuda')

    for stride in [1, 2]:
        for dilation in [1, 2]:
            for padding in [0, 1, 128 // 2]:
                if (stride, padding, dilation) in [(1, 1, 2), (2, 64, 2)]:
                    continue
                t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
                y = mnm.conv2d(x, w, stride=stride, dilation=dilation, padding=padding)
                np.testing.assert_allclose(y.asnumpy(), t_y.numpy(), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_mnm_conv2d()
