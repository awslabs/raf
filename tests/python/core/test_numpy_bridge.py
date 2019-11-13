import numpy as np
import pytest

import mnm


def test_mnm_array_cpu():
    array = mnm.array([1, 2, 3], dtype="int8", ctx="cpu")
    array = array.asnumpy()
    assert np.all(array == [1, 2, 3])


@pytest.mark.skipif(mnm._ffi.build_info.use_cuda() == "OFF", reason="CUDA is not enabled")  # pylint: disable=protected-access
def test_mnm_array_cuda():
    array = mnm.array([1, 2, 3], dtype="int8", ctx="cuda")
    array = array.asnumpy()
    assert np.all(array == [1, 2, 3])


if __name__ == "__main__":
    test_mnm_array_cpu()
    test_mnm_array_cuda()
