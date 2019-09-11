import numpy as np
import torch.nn.functional as F
import torch
import pytest

import mnm


def test_conv2d():

    img_ = np.random.randn(1, 3, 128, 128).astype('float32')
    ker_ = np.random.randn(128, 3, 6, 6).astype('float32')

    img = mnm.array(img_, dtype='float32', ctx='cuda')
    ker = mnm.array(ker_, dtype='float32', ctx='cuda')
    for stride in [1, 2]:
        for dilation in [1, 2]:
            for padding in [0, 1, 128 // 2]:
                if (stride, padding, dilation) in [(1, 1, 2), (2, 64, 2)]:
                    continue
                ref = F.conv2d(torch.Tensor(img_), torch.Tensor(ker_),
                               stride=stride, dilation=dilation, padding=padding)
                out = mnm.nn.conv2d(img, ker, stride=stride,
                                    dilation=dilation, padding=padding)
                np.testing.assert_allclose(
                    out.asnumpy(), ref.numpy(), rtol=1e-4, atol=1e-4)


def test_pool2d():

    img_ = np.random.randn(1, 128, 128, 128).astype('float32')
    img = mnm.array(img_, dtype='float32', ctx='cuda')

    out = mnm.nn.avg_pool2d(img, kernel=3, stride=1)
    ref = F.avg_pool2d(torch.Tensor(img_), kernel_size=3, stride=1)
    np.testing.assert_allclose(
        out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)

    out = mnm.nn.max_pool2d(img, kernel=3, stride=1)
    ref = F.max_pool2d(torch.Tensor(img_), kernel_size=3, stride=1)
    np.testing.assert_allclose(
        out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_relu():
    img_ = np.random.randn(5).astype('float32')
    img = mnm.array(img_, dtype='float32', ctx='cuda')

    out = mnm.nn.relu(img)
    ref = torch.relu(torch.Tensor(img_))
    np.testing.assert_allclose(
        out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_tanh():
    img_ = np.random.randn(5).astype('float32')
    img = mnm.array(img_, dtype='float32', ctx='cuda')

    out = mnm.nn.tanh(img)
    ref = torch.tanh(torch.Tensor(img_))
    np.testing.assert_allclose(
        out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_sigmoid():
    img_ = np.random.randn(5).astype('float32')
    img = mnm.array(img_, dtype='float32', ctx='cuda')

    out = mnm.nn.sigmoid(img)
    ref = torch.sigmoid(torch.Tensor(img_))
    np.testing.assert_allclose(
        out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_softmax():
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    shapes = [
            [2, 2, 2, 2],
            [2, 2, 2],
            [2, 2],
            [2, 3, 4, 5],
    ]

    for shape in shapes:

        img_ = np.random.randn(*shape).astype('float32')
        img = mnm.array(img_, dtype='float32', ctx='cuda')

        for i in range(-len(shape), len(shape)):
            out = mnm.nn.softmax(img, axis=i)
            ref = F.softmax(torch.Tensor(img_), dim=i)
            np.testing.assert_allclose(
                out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_log_softmax():
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    shapes = [
            [2, 2, 2, 2],
            [2, 2, 2],
            [2, 2],
            [2, 3, 4, 5],
    ]

    for shape in shapes:

        img_ = np.random.randn(*shape).astype('float32')
        img = mnm.array(img_, dtype='float32', ctx='cuda')

        for i in range(-len(shape), len(shape)):
            out = mnm.nn.log_softmax(img, axis=i)
            ref = F.log_softmax(torch.Tensor(img_), dim=i)
            np.testing.assert_allclose(
                out.asnumpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


def test_batch_flatten():
    img_ = np.random.randn(128, 2, 2, 2).astype('float32')
    out_ = mnm.nn.batch_flatten(mnm.array(img_, dtype='float32', ctx='cpu'))
    out = out_.asnumpy()
    assert(out.shape == (128, 8))


def test_batchnorm():
    x_ = np.random.randn(8, 8, 8, 8).astype('float32')
    mean_ = np.random.randn(8).astype('float32')
    vari_ = np.random.randn(8).astype('float32')
    scal_ = np.random.randn(8).astype('float32')
    bias_ = np.random.randn(8).astype('float32')

    out_ = mnm.nn.batch_norm2d(
        mnm.array(x_, dtype='float32', ctx='cuda'),
        mnm.array(mean_, dtype='float32', ctx='cuda'),
        mnm.array(vari_, dtype='float32', ctx='cuda'),
        mnm.array(scal_, dtype='float32', ctx='cuda'),
        mnm.array(bias_, dtype='float32', ctx='cuda'),)

    ref = F.batch_norm(
            torch.Tensor(x_),
            torch.Tensor(mean_),
            torch.Tensor(vari_),
            torch.Tensor(scal_),
            torch.Tensor(bias_),)

    np.testing.assert_allclose(out_.asnumpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

    out_ = mnm.nn.batch_norm2d(
        mnm.array(x_, dtype='float32', ctx='cuda'),
        mnm.array(mean_, dtype='float32', ctx='cuda'),
        mnm.array(vari_, dtype='float32', ctx='cuda'),
        mnm.array(scal_, dtype='float32', ctx='cuda'),
        mnm.array(bias_, dtype='float32', ctx='cuda'),
        is_training=True)

    ref = F.batch_norm(
            torch.Tensor(x_),
            torch.Tensor(mean_),
            torch.Tensor(vari_),
            torch.Tensor(scal_),
            torch.Tensor(bias_),
            training=True)

    np.testing.assert_allclose(out_.asnumpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
    test_conv2d()
    test_pool2d()
    test_relu()
    test_sigmoid()
    test_tanh()
    test_softmax()
    test_log_softmax()
    test_batch_flatten()
    test_batchnorm()
