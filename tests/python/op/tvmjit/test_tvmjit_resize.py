import pytest
from torch.nn.functional import interpolate

import mnm
from mnm.testing import get_device_list, randn_torch, check, with_seed

@pytest.mark.parametrize("params", [
    {"batchs": 1, "layout": "NCHW", "orig_shape": (718, 718),
     "to_shape": (64, 64), "infer_shape": (1, 3, 64, 64)},
    {"batchs": 32, "layout": "NCHW", "orig_shape": (32, 32),
     "to_shape": (64, 32), "infer_shape": (32, 3, 64, 32)},
    {"batchs": 32, "layout": "NCHW", "orig_shape": (32, 32),
     "to_shape": 64, "infer_shape": (32, 3, 64, 64)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (718, 718),
     "to_shape": (64, 64), "infer_shape": (32, 64, 64, 3)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (32, 32),
     "to_shape": (64, 32), "infer_shape": (32, 64, 32, 3)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (32, 32),
     "to_shape": 64, "infer_shape": (32, 64, 64, 3)},
])
@pytest.mark.parametrize("in_dtype", ["float32", "float64"])
@pytest.mark.parametrize("out_dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", get_device_list())
@with_seed(0)
def test_resize(device, params, in_dtype, out_dtype):
    batchs, layout, orig_shape, to_shape, infer_shape = \
        params["batchs"], params["layout"], params["orig_shape"], \
        params["to_shape"], params["infer_shape"]

    # PyTorch only support NCHW, so for NHWC, only compared the shape
    if layout == "NHWC":
        m_x, _ = randn_torch((batchs, orig_shape[0], orig_shape[1], 3),
                             dtype=in_dtype, device=device)
        m_y = mnm.resize(m_x, to_shape, layout, out_dtype=out_dtype)
    elif layout == "NCHW":
        m_x, t_x = randn_torch((batchs, 3, orig_shape[0], orig_shape[1]),
                               dtype=in_dtype, device=device)
        m_y = mnm.resize(m_x, to_shape, layout, "nearest_neighbor", \
                            "asymmetric", out_dtype=out_dtype)
        t_y = interpolate(t_x, to_shape, mode="nearest")
        check(m_y, t_y, rtol=1e-4, atol=1e-4)

    assert m_y.shape == infer_shape


if __name__ == "__main__":
    pytest.main([__file__])
