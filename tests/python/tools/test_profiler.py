# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

import raf
from raf._op import sym
from raf.utils import profiler
from raf.testing import randn


class TestNet(raf.Model):
    def build(self, axes=None):
        self._axes = axes  # pylint: disable=attribute-defined-outside-init

    @raf.model.trace
    def forward(self, x, y_true):
        y_pred = self.forward_infer(x)
        loss = sym.nll_loss(y_true, y_pred)
        return loss

    @raf.model.trace
    def forward_infer(self, x):
        x = sym.transpose(x, self._axes)
        x = sym.ceil(x)
        x = sym.cos(x)
        x = sym.floor(x)
        x = sym.transpose(x, self._axes)
        return x


class TestCuda(raf.Model):
    def build(self):
        pass

    @raf.model.trace
    def forward(self, m_a, m_b):  # pylint: disable=no-self-use
        return raf.matmul(m_a, m_b)


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("i", [0])
def test_profiler_with_cuda(i):
    profiler.start()
    device = "cuda({})".format(i)
    print("device:", device)
    n = 4
    k = 8
    m = 4
    m_x, _ = randn((n, k))
    m_y, _ = randn((k, m))
    model = TestCuda()
    model.to(device=device)
    print("### Switch to training mode")
    model.train_mode()
    for epoch in range(5):
        loss = model(m_x, m_y)
        print("epoch", epoch, ": Loss: ", loss)
        loss.backward()
        profiler.stop()
    data = profiler.get()
    assert len(data["traceEvents"]) >= 0
    op_count = 0
    for e in data["traceEvents"]:
        if e["name"] == "raf.op.matmul":
            op_count += 1
    assert op_count > 0


@pytest.mark.parametrize("i", [0])
def test_profiler_without_cuda(i):
    profiler.start()
    batch_size = 16
    device = "cpu({})".format(i)
    features = 4
    x = np.arange(features * batch_size).reshape(batch_size, features)
    y = np.random.randint(0, features, size=batch_size)
    m_x = raf.array(x, dtype="float32", device=device, name="cck-m_x")
    m_y = raf.array(y, device=device, name="cck-m_y")
    model = TestNet((0, 1))
    print("### Switch to training mode")
    model.train_mode()
    for epoch in range(5):
        loss = model(m_x, m_y)
        print("epoch", epoch, ": Loss: ", loss)
        loss.backward()
        profiler.stop()
    data = profiler.get()
    assert len(data["traceEvents"]) >= 0
    op_count = 0
    for e in data["traceEvents"]:
        if e["name"] == "raf.op.transpose":
            op_count += 1
    assert op_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
