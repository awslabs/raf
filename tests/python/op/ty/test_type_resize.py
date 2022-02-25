# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
from raf._ffi.pass_ import InferType
from raf.testing import check_type, randn
from tvm.relay import TensorType, FuncType

# pylint: disable=too-many-locals, import-outside-toplevel, attribute-defined-outside-init
@pytest.mark.parametrize(
    "params",
    [
        {
            "batchs": 32,
            "layout": "NCHW",
            "orig_shape": (718, 718),
            "to_shape": (64, 64),
            "infer_shape": (32, 3, 64, 64),
        },
        {
            "batchs": 32,
            "layout": "NCHW",
            "orig_shape": (32, 32),
            "to_shape": 400,
            "infer_shape": (32, 3, 400, 400),
        },
        {
            "batchs": 32,
            "layout": "NHWC",
            "orig_shape": (718, 718),
            "to_shape": (64, 64),
            "infer_shape": (32, 64, 64, 3),
        },
        {
            "batchs": 32,
            "layout": "NHWC",
            "orig_shape": (32, 32),
            "to_shape": 400,
            "infer_shape": (32, 400, 400, 3),
        },
    ],
)
@pytest.mark.parametrize("in_dtype", ["float32"])
@pytest.mark.parametrize("out_dtype", ["float32"])
def test_resize2d(params, in_dtype, out_dtype):
    class Resize2D(raf.Model):
        def build(self, to_size, layout, out_dtype):
            self._size = to_size
            self._layout = layout
            self._dtype = out_dtype

        @raf.model.trace
        def forward(self, x):
            return raf.resize2d(x, self._size, self._layout, out_dtype=self._dtype)

    batchs, layout, orig_shape, to_shape, infer_shape = (
        params["batchs"],
        params["layout"],
        params["orig_shape"],
        params["to_shape"],
        params["infer_shape"],
    )

    if layout == "NCHW":
        shape = [batchs, 3, orig_shape[0], orig_shape[1]]
    elif layout == "NHWC":
        shape = [batchs, orig_shape[0], orig_shape[1], 3]

    # forward
    model = Resize2D(to_shape, layout, out_dtype)
    m_x, _ = randn(shape, dtype=in_dtype)
    record = model._internal(m_x)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    x_ty = TensorType(shape, dtype=in_dtype)
    y_ty = TensorType(infer_shape, dtype=out_dtype)
    expected_type = FuncType([x_ty], y_ty)
    check_type(m_mod["main"], expected_type)


if __name__ == "__main__":
    pytest.main([__file__])
