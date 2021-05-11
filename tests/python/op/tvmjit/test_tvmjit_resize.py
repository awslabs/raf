import pytest
import numpy as np
from torch import tensor
from torch.nn.functional import interpolate

import mnm

@pytest.mark.parametrize("params", [
    {"batchs": 1, "layout": "NCHW", "orig_shape": (718, 718),
     "to_shape": (64, 64), "infer_shape": (1, 3, 64, 64)},
    {"batchs": 32, "layout": "NCHW", "orig_shape": (32, 32),
     "to_shape": (400, 400), "infer_shape": (32, 3, 400, 400)},
    {"batchs": 32, "layout": "NCHW", "orig_shape": (32, 32),
     "to_shape": (64, 32), "infer_shape": (32, 3, 64, 32)},
    {"batchs": 32, "layout": "NCHW", "orig_shape": (32, 32),
     "to_shape": 64, "infer_shape": (32, 3, 64, 64)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (718, 718),
     "to_shape": (64, 64), "infer_shape": (32, 64, 64, 3)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (32, 32),
     "to_shape": (400, 400), "infer_shape": (32, 400, 400, 3)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (32, 32),
     "to_shape": (64, 32), "infer_shape": (32, 64, 32, 3)},
    {"batchs": 32, "layout": "NHWC", "orig_shape": (32, 32),
     "to_shape": 64, "infer_shape": (32, 64, 64, 3)},
])
@pytest.mark.parametrize("in_dtype", ["float32", "float64"])
@pytest.mark.parametrize("out_dtype", ["float32", "float64"])
def test_resize(params, in_dtype, out_dtype):
    batchs, layout, orig_shape, to_shape, infer_shape = \
        params["batchs"], params["layout"], params["orig_shape"], \
        params["to_shape"], params["infer_shape"]

    # PyTorch only support NCHW, so for NHWC, only compared the shape
    if layout == "NHWC":
        m_x = mnm.array(np.random.randn(batchs, orig_shape[0], orig_shape[1], 3).astype(in_dtype))
        m_y = mnm.resize(m_x, to_shape, layout, out_dtype=out_dtype)
        assert m_y.shape == infer_shape
    elif layout == "NCHW":
        data = np.random.randn(batchs, 3, orig_shape[0], orig_shape[1]).astype(in_dtype)
        m_x = mnm.array(data)
        t_x = tensor(data) # pylint: disable=not-callable
        m_y = mnm.resize(m_x, to_shape, layout, "nearest_neighbor", \
                            "asymmetric", out_dtype=out_dtype)
        t_y = interpolate(t_x, to_shape, mode="nearest")

        # convert all to ndarray and make sure they have the same precision
        m_y = np.around(m_y.asnumpy(), 2)
        t_y = np.around(t_y.numpy().astype(out_dtype), 2)
        # use diff to compare the result from pytorch and the one from meta,
        # due to instable floating point value
        diff = abs((m_y - t_y).sum()/m_y.sum())
        assert m_y.shape == infer_shape
        assert diff < 0.0001




if __name__ == "__main__":
    pytest.main([__file__])
