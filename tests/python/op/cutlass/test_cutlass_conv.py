# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-member,no-self-use
import pytest
import torch
import torch.nn.functional as F

import raf
from raf.testing import randn_torch, run_vm_model, check, DialectChecker


def verify_ir(mod):
    with raf.device("cuda"):
        mod = raf._ffi.pass_.ToGraphNormalForm()(mod)
        mod = raf._ffi.pass_.ToBasicBlockNormalForm()(mod)
        mod = raf._ffi.pass_.FuseDialect()(mod)
        DialectChecker("cutlass").visit(mod["main"])


@pytest.mark.skipif(not raf.build.with_cutlass(), reason="CUTLASS is not enabled")
@pytest.mark.parametrize(
    "shapes",
    [
        ((4, 256, 32, 32), (64, 256, 1, 1)),
        ((8, 3, 32, 32), (16, 3, 3, 3)),
    ],
)
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv2d_relu(shapes, stride, dilation, padding):
    device, dtype = "cuda", "float32"

    class Conv2D(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            y = raf.conv2d(
                x,
                w,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            y = raf.relu(y)
            return y

    xshape, wshape = shapes
    m_x, t_x = randn_torch(xshape, device=device, dtype=dtype)
    m_w, t_w = randn_torch(wshape, device=device, dtype=dtype)
    m_x = raf.transpose(m_x, (0, 2, 3, 1))
    m_w = raf.transpose(m_w, (0, 2, 3, 1))
    model = Conv2D()
    mod = model._internal(m_x, m_w).mod
    verify_ir(mod)
    m_y = run_vm_model(model, device, [m_x, m_w])
    t_model = (
        torch.nn.Conv2d(
            wshape[3],
            wshape[0],
            (wshape[1], wshape[2]),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        .cuda()
        .float()
    )
    t_model.weight = torch.nn.Parameter(t_w)
    t_model = t_model.to(memory_format=torch.channels_last)
    t_y = t_model(t_x)
    t_y = F.relu(t_y.permute(0, 2, 3, 1))
    check(m_y, t_y, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
