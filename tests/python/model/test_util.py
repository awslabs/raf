# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=attribute-defined-outside-init, no-self-use
import pytest
import numpy as np

import raf
from raf._core.ndarray import ndarray
from raf.model.model import calc_model_gflops, get_param_size
from raf.testing import check, randn


def test_param_size():
    class Model(raf.Model):
        def build(self):
            self.attn = ndarray(
                np.ones(((1, 512)), dtype="int64"),
                name="attn",
                device="cpu",
            )
            self.weight = ndarray(
                np.ones(((128, 4)), dtype="float32"),
                name="weight",
                device="cpu",
            )

        @raf.model.trace
        def forward(self, data):
            a_1 = raf.strided_slice(self.attn, (0, 0), (1, 128), (1, 1), "end")
            a_2 = raf.embedding(data, a_1)
            a_3 = raf.matmul(a_2, self.weight)
            return a_3

    model = Model()

    # Infer mode should return 0
    assert get_param_size(model) == 0

    # Parameter size should be 128 * 4 (excluding attn with int64 dtype).
    model.train_mode()
    assert get_param_size(model) == 512


def test_calc_gflops():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, data, weight):
            a_1 = raf.matmul(data, weight)  # 2 * (8 * 16 * 4) = 1024
            a_2 = raf.add(a_1, a_1)  # 8 * 4 = 32
            return a_2

    m_x, _ = randn((8, 16), requires_grad=True, device="cpu")
    m_w, _ = randn((16, 4), requires_grad=True, device="cpu")
    gflops = calc_model_gflops(Model(), "cpu", [m_x, m_w])
    check(gflops, 1056 / 1e9, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
