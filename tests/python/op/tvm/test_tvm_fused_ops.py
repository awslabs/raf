# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import raf
from raf.testing import get_testable_devices, run_vm_model


@pytest.mark.parametrize("device", get_testable_devices())
def test_avg_pool2d_dx_fuse_relu_dx(device):
    # pylint: disable=attribute-defined-outside-init, unnecessary-pass
    class AvgPoolDxReLUDx(raf.Model):
        def build(self, y, dy, relu_x, relu_dy):
            self.y = y
            self.dy = dy
            self.relu_x = relu_x
            self.relu_dy = relu_dy
            pass

        @raf.model.trace
        def forward(self, x):
            pooldx = raf.avg_pool2d_dx(
                x,
                self.y,
                self.dy,
                kernel=3,
                stride=1,
                padding=0,
                dilation=1,
                ceil_mode=False,
                include_pad=True,
            )
            out = raf.relu_dx(pooldx, self.relu_x, self.relu_dy)
            return out

    x = raf.array(np.random.randn(8, 3, 32, 32), dtype="float64", device=device)
    relu_x = raf.relu(x)
    relu_dy = raf.array(np.random.randn(*relu_x.shape), dtype="float64", device=device)
    y = raf.avg_pool2d(
        relu_x, kernel=3, stride=1, padding=0, dilation=1, ceil_mode=False, include_pad=True
    )
    dy = raf.array(np.random.randn(*y.shape), dtype="float64", device=device)
    model = AvgPoolDxReLUDx(y, dy, relu_x, relu_dy)
    run_vm_model(model, device, [x])


if __name__ == "__main__":
    pytest.main([__file__])
