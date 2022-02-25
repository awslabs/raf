# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init
import sys
import numpy as np
import pytest
from tvm import relay
import raf
from raf.testing import run_vm_model
from raf.model.trace import trace_mutate_attr


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("shape", [(4, 4), (100,)])
def test_threefry_generate(device, shape):
    class ThreefryGenerate(raf.Model):
        def build(self, key, shape):
            self.key = key
            self.shape = shape

        @raf.model.trace
        def forward(self):
            res = raf.threefry_generate(self.key, self.shape)
            trace_mutate_attr(self, "key", res[0])
            return res[1]

    seed = np.random.randint(0, high=sys.maxsize)
    key = raf.array(relay.random.threefry_key(seed).data.numpy(), dtype="uint64", device=device)
    model = ThreefryGenerate(key, shape)
    m_y0 = model()
    for _ in range(10):
        m_y = model()
        if m_y == m_y0:
            raise ValueError("Random state not updated.")
    v_y0 = run_vm_model(model, device, [])
    for _ in range(10):
        v_y = run_vm_model(model, device, [])
        if v_y == v_y0:
            raise ValueError("Random state not updated.")


if __name__ == "__main__":
    pytest.main([__file__])
