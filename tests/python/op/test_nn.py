import numpy as np
import torch

import mnm

def test_mnm_conv2d():

    img_ = np.random.randn(1, 3, 128, 128).astype('float32')
    ker_ = np.random.randn(128, 3, 3, 3).astype('float32')

    img = mnm.array(img_, dtype='float32', ctx='cuda')
    ker = mnm.array(ker_, dtype='float32', ctx='cuda')
    for stride in [1, 2]:
        for dilation in [1, 2]:
            for padding in [0, 1, 128 // 2]:
                if (stride, padding, dilation) in [(1, 1, 2), (2, 64, 2)]:
                    continue
                ref = torch.nn.functional.conv2d(torch.Tensor(img_), torch.Tensor(ker_),
                        stride=stride, dilation=dilation, padding=padding)
                out = mnm.nn.conv2d(img, ker,
                        stride=stride, dilation=dilation, padding=padding)
                np.testing.assert_allclose(out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    test_mnm_conv2d()
