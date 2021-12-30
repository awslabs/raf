import numpy as np
import pytest

import mnm
from mnm.testing import get_testable_devices, run_vm_model


@pytest.mark.parametrize("device", get_testable_devices())
def test_avg_pool2d_dx_fuse_relu_dx(device):
    # pylint: disable=attribute-defined-outside-init, unnecessary-pass
    class AvgPoolDxReLUDx(mnm.Model):
        def build(self, y, dy, relu_x, relu_dy):
            self.y = y
            self.dy = dy
            self.relu_x = relu_x
            self.relu_dy = relu_dy
            pass

        @mnm.model.trace
        def forward(self, x):
            pooldx = mnm.avg_pool2d_dx(
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
            out = mnm.relu_dx(pooldx, self.relu_x, self.relu_dy)
            return out

    x = mnm.array(np.random.randn(8, 3, 32, 32), dtype="float64", device=device)
    relu_x = mnm.relu(x)
    relu_dy = mnm.array(np.random.randn(*relu_x.shape), dtype="float64", device=device)
    y = mnm.avg_pool2d(
        relu_x, kernel=3, stride=1, padding=0, dilation=1, ceil_mode=False, include_pad=True
    )
    dy = mnm.array(np.random.randn(*y.shape), dtype="float64", device=device)
    model = AvgPoolDxReLUDx(y, dy, relu_x, relu_dy)
    run_vm_model(model, device, [x])


if __name__ == "__main__":
    pytest.main([__file__])
