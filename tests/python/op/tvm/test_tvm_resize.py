# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.nn.functional import interpolate

import raf
from raf.testing import get_testable_devices, randn_torch, check, with_seed


@pytest.mark.parametrize(
    "params",
    [
        {
            "batchs": 32,
            "layout": "NCHW",
            "orig_shape": (32, 32),
            "to_shape": (64, 32),
            "infer_shape": (32, 3, 64, 32),
        },
        {
            "batchs": 32,
            "layout": "NCHW",
            "orig_shape": (32, 32),
            "to_shape": 64,
            "infer_shape": (32, 3, 64, 64),
        },
        {
            "batchs": 32,
            "layout": "NHWC",
            "orig_shape": (32, 32),
            "to_shape": (64, 32),
            "infer_shape": (32, 64, 32, 3),
        },
        {
            "batchs": 32,
            "layout": "NHWC",
            "orig_shape": (32, 32),
            "to_shape": 64,
            "infer_shape": (32, 64, 64, 3),
        },
    ],
)
@pytest.mark.parametrize("device", get_testable_devices())
@pytest.mark.parametrize("method", ["nearest_neighbor", "linear", "cubic"])
@with_seed(0)
def test_resize2d(device, params, method):
    # pylint: disable=no-member, too-many-arguments, too-many-locals
    batchs, layout, orig_shape, to_shape, infer_shape = (
        params["batchs"],
        params["layout"],
        params["orig_shape"],
        params["to_shape"],
        params["infer_shape"],
    )
    # Skip float64 tests since it may not be supported by te.Gradient
    in_dtype = "float32"
    out_dtype = "float32"
    # PyTorch only support NCHW, so for NHWC, only compared the shape
    if layout == "NHWC":
        m_x, _ = randn_torch(
            (batchs, orig_shape[0], orig_shape[1], 3), dtype=in_dtype, device=device
        )
        m_y = raf.resize2d(m_x, to_shape, layout, out_dtype=out_dtype)
    elif layout == "NCHW":
        # test forward
        m_x, t_x = randn_torch(
            (batchs, 3, orig_shape[0], orig_shape[1]),
            dtype=in_dtype,
            device=device,
            requires_grad=True,
        )
        coord_trans_mode = "asymmetric"
        torch_mode = "nearest"
        if method == "linear":
            coord_trans_mode = "half_pixel"
            torch_mode = "bilinear"
        elif method == "cubic":
            coord_trans_mode = "half_pixel"
            torch_mode = "bicubic"
        m_y = raf.resize2d(
            m_x,
            to_shape,
            layout,
            method=method,
            coordinate_transformation_mode=coord_trans_mode,
            cubic_alpha=-0.75,
            out_dtype=out_dtype,
        )
        t_y = interpolate(t_x, to_shape, mode=torch_mode)
        check(m_y, t_y, rtol=1e-5, atol=1e-5)

        # test backward
        m_dy, t_dy = randn_torch(t_y.shape, dtype=out_dtype, device=device)
        m_y.backward(m_dy)
        t_y.backward(t_dy)
        check(m_x.grad, t_x.grad, rtol=1e-5, atol=1e-5)

    assert m_y.shape == infer_shape


if __name__ == "__main__":
    pytest.main([__file__])
