# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
import pytest
import numpy as np

import raf
from raf.testing import check, with_dialect
from raf.testing.utils import run_model


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [(64, 128), (128, 256)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tensor_fusion_and_defusion(shape, dtype):
    size = 1
    for axis in shape:
        size *= axis
    sizes = [size, size]
    tuple_shape = shape + shape
    shape_indices = [len(shape), 2 * len(shape)]

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x_1 = [x, x]
            x_2 = raf.fuse_tensor(x_1)
            x_3 = raf.defuse_tensor(x_2, sizes, tuple_shape, shape_indices)
            return x_3[0]

    x = np.ones(shape, dtype=dtype)
    x = raf.array(x, device="cuda(0)")
    model = Model()
    device = f"cuda(0)"
    y_c = run_model(model, [x], device)
    rtol = 1e-4 if dtype == "float32" else 4e-2
    atol = 1e-4 if dtype == "float32" else 4e-2
    check(x, y_c, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
