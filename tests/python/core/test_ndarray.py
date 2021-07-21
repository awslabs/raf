import numpy as np
import pytest

import mnm


def test_requires_grad():
    a = mnm.array([1, 2, 3], dtype="float32")
    assert not a.requires_grad
    bools = [False, True, True, False, True, False, False, False, True]
    for val in bools:
        a.requires_grad = val
        assert a.requires_grad == val


def test_mutation():
    a = mnm.array([1, 2, 3], dtype="float32")
    a[:] = np.array([4, 5, 6], dtype="float64")
    assert a.shape == (3, )
    assert a.dtype == "float32"
    np.testing.assert_allclose(np.array([4, 5, 6], dtype='float32'), a.numpy())


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_move_device():
    a = mnm.array([1, 2, 3], dtype='float32')
    a = a.to(device='cuda')
    assert a.device.startswith('cuda')
    np.testing.assert_allclose(np.array([1, 2, 3], dtype='float32'), a.numpy())


if __name__ == "__main__":
    test_requires_grad()
    test_mutation()
    test_move_device()
