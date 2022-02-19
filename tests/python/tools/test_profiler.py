# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import numpy as np

import mnm
from mnm._op import sym
from mnm.utils import profiler
from mnm.testing import randn


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
        if e["name"] == "mnm.op.matmul":
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
    m_x = mnm.array(x, dtype="float32", device=device, name="cck-m_x")
    m_y = mnm.array(y, device=device, name="cck-m_y")
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
        if e["name"] == "mnm.op.transpose":
            op_count += 1
    assert op_count > 0


if __name__ == "__main__":
    pytest.main([__file__])
