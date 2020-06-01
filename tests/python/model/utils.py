import numpy as np
import torch

import mnm

def check(m_x, t_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    t_x = t_x.detach().cpu().numpy()
    np.testing.assert_allclose(m_x, t_x, rtol=rtol, atol=atol)

def one_hot(batch_size, num_classes, ctx="cuda", dtype="float32"):
    targets = np.random.randint(0, num_classes, size=batch_size)
    m_x = np.zeros([batch_size, num_classes], dtype=dtype)
    m_x[range(batch_size), targets] = 1
    m_x = mnm.array(m_x, ctx=ctx)
    t_x = torch.tensor(targets, requires_grad=False)  # pylint: disable=not-callable
    assert list(m_x.shape) == [batch_size, num_classes]
    assert list(t_x.shape) == [batch_size]
    return m_x, t_x

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
    t_x = torch.tensor(x, requires_grad=requires_grad)  # pylint: disable=not-callable
    return m_x, t_x

def t2m_param(param, ctx="cuda"):
    return mnm.ndarray(param.detach().numpy(), ctx=ctx)  # pylint: disable=unexpected-keyword-arg
