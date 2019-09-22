import numpy as np

import mnm


def test_copy():
    np_a = np.random.randn(2, 5).astype('float32')
    a = mnm.array(np_a, dtype="float32", ctx="cuda(0)")
    b = mnm.copy(a)
    np_b = b.asnumpy()
    np.testing.assert_allclose(np_a, np_b, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    test_copy()
