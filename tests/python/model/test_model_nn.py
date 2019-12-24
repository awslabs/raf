import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mnm


def randn(shape, *, ctx="cuda", dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    t_x = torch.tensor(x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def randn_pos(shape, *, ctx="cuda", dtype="float32", std=1.0):
    x = np.abs(np.random.randn(*shape)) * std + 1e-5
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    t_x = torch.tensor(x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x


def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)


# TODO(@were): allow affine=False
@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("num_features", [1, 16])
@pytest.mark.parametrize("affine", [True])
@pytest.mark.parametrize("is_train", [False, True])
def test_model_batch_norm(num_features, affine, is_train):
    m_x, t_x = randn([8, num_features, 5, 6])
    m_m, _ = randn([num_features])
    m_v, _ = randn_pos([num_features])
    if affine:
        m_w, _ = randn([num_features])
        m_b, _ = randn_pos([num_features])
    model = mnm.model.nn.BatchNorm(num_features=num_features, affine=affine)
    model.running_mean = m_m
    model.running_var = m_v
    if affine:
        model.w = m_w
        model.b = m_b
    # pylint: disable=no-member
    t_model = torch.nn.BatchNorm2d(num_features=num_features, affine=affine)
    t_model.running_mean[:] = torch.from_numpy(m_m.asnumpy())
    t_model.running_var[:] = torch.from_numpy(m_v.asnumpy())
    if affine:
        t_model.weight[:] = torch.from_numpy(m_w.asnumpy())
        t_model.bias[:] = torch.from_numpy(m_b.asnumpy())
    # pylint: enable=no-member
    if is_train:
        model.train_mode()
        t_model.train()
    else:
        model.infer_mode()
        t_model.eval()
    m_y = model(m_x)
    t_y = t_model(t_x)
    check(m_y, t_y)
    check(model.running_mean, t_model.running_mean)
    check(model.running_var, t_model.running_var)
    if affine:
        check(model.w, t_model.weight)
        check(model.b, t_model.bias)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("bias", [False, True])
def test_model_conv2d(stride, dilation, padding, bias):
    m_x, t_x = randn([8, 3, 32, 32], std=0.001)
    m_w, t_w = randn([16, 3, 3, 3], std=0.01)
    if bias:
        m_b, t_b = randn([16], std=0.001)
    model = mnm.model.Conv2d(in_channels=3,
                             out_channels=16,
                             kernel_size=3,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=1,
                             bias=bias)
    model.w = m_w
    if bias:
        model.b = m_b
    m_y = model(m_x)
    t_y = F.conv2d(t_x, t_w, stride=stride, dilation=dilation, padding=padding)
    if bias:
        t_b = t_b.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=0)
        t_y = t_y + t_b
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("in_features", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("out_features", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("bias", [False, True])
def test_model_dense(batch_size, in_features, out_features, bias):
    m_x, t_x = randn([batch_size, in_features])
    m_w, _ = randn([out_features, in_features])
    if bias:
        m_b, _ = randn([out_features])

    # pylint: disable=no-member
    t_model = torch.nn.Linear(in_features, out_features, bias=bias)
    t_model.weight[:] = torch.from_numpy(m_w.asnumpy())
    if bias:
        t_model.bias[:] = torch.from_numpy(m_b.asnumpy())
    # pylint: enable=no-member
    model = mnm.model.Linear(in_features=in_features,
                             out_features=out_features,
                             bias=bias)
    model.w = m_w
    if bias:
        model.b = m_b
    m_y = model(m_x)
    t_y = t_model(t_x)
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


def _fake_test_relu():
    m_x, _ = randn([1, 2, 3, 4], ctx="cpu")

    class ReLU(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm.relu(x)

    model = ReLU()
    model(m_x)


def _fake_test_conv2d():
    model = mnm.model.Conv2d(in_channels=3,
                             out_channels=16,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             dilation=1,
                             groups=1,
                             bias=False)
    m_x, _ = randn([8, 3, 32, 32], ctx="cpu")
    model(m_x)


def _fake_test_batch_norm():
    num_features = 128
    model = mnm.model.BatchNorm(num_features=num_features,
                                eps=1e-5,
                                momentum=0.1,
                                affine=True)
    m_x, _ = randn([5, num_features, 3, 3], ctx="cpu")
    model(m_x)


def _fake_test_binary_add():
    m_x1, _ = randn([1, 2, 3], ctx="cpu")
    m_x2, _ = randn([2, 2, 3], ctx="cpu")

    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x1, x2):  # pylint: disable=no-self-use
            return mnm.add(x1, x2)

    model = Add()
    model(m_x1, m_x2)


if __name__ == "__main__":
    # _fake_test_relu()
    # _fake_test_conv2d()
    # _fake_test_batch_norm()
    # _fake_test_binary_add()
    pytest.main([__file__])
