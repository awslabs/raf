import pytest
import numpy as np

import mnm
from mnm._op import sym
from mnm.utils import profiler


def randn(shape, *, ctx="cuda", dtype="float32", std=1.0):
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    x = x.astype(dtype)
    m_x = mnm.array(x, ctx=ctx)
    return m_x


class TestNet(mnm.Model):
    def build(self, axes=None):
        self._axes = axes  # pylint: disable=attribute-defined-outside-init

    @mnm.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        loss = sym.nll_loss(y_true, y_pred)
        return loss

    @mnm.model.trace
    def forward_infer(self, x):
        x = sym.transpose(x, self._axes)
        x = sym.ceil(x)
        x = sym.cos(x)
        x = sym.floor(x)
        x = sym.transpose(x, self._axes)
        return x


class TestCuda(mnm.Model):
    def build(self):
        pass

    @mnm.model.trace
    def forward(self, m_a, m_b):  # pylint: disable=no-self-use
        return mnm.matmul(m_a, m_b)


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("i", [0])
def test_profiler_with_cuda(i):
    profiler.start()
    ctx = "cuda({})".format(i)
    print("ctx:", ctx)
    n = 4
    k = 8
    m = 4
    m_x = randn((n, k))
    m_y = randn((k, m))
    model = TestCuda()
    model.to(ctx=ctx)
    print("### Switch to training mode")
    model.train_mode()
    for epoch in range(5):
        loss = model(m_x, m_y)
        print("epoch", epoch, ": Loss: ", loss)
        loss.backward()
        profiler.stop()
    data = profiler.get()
    assert len(data['traceEvents']) >= 0
    op_count = 0
    for e in data['traceEvents']:
        if e['name'] == 'mnm.op.matmul':
            op_count += 1
    assert op_count > 0


@pytest.mark.parametrize("i", [0])
def test_profiler_without_cuda(i):
    profiler.start()
    batch_size = 16
    ctx = "cpu({})".format(i)
    features = 4
    x = np.arange(features*batch_size).reshape(batch_size, features)
    y = np.arange(features*batch_size).reshape(batch_size, features)
    m_x = mnm.array(x, dtype="float32", ctx=ctx, name='cck-m_x')
    m_y = mnm.array(y, dtype='float32', ctx=ctx, name='cck-m_y')
    model = TestNet((0, 1))
    print("### Switch to training mode")
    model.train_mode()
    for epoch in range(5):
        loss = model(m_x, m_y)
        print("epoch", epoch, ": Loss: ", loss)
        loss.backward()
        profiler.stop()
    data = profiler.get()
    assert len(data['traceEvents']) >= 0
    op_count = 0
    for e in data['traceEvents']:
        if e['name'] == 'mnm.op.transpose':
            op_count += 1
    assert op_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
