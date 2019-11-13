import numpy as np
import pytest

import mnm


def test_batch_flatten_0d():
    x = np.random.randn(1).astype("float32").reshape(())
    x = mnm.array(x)
    assert x.shape == ()
    with pytest.raises(ValueError):
        mnm.batch_flatten(x)


def test_batch_flatten_1d():
    x = np.random.randn(5).astype("float32")
    x = mnm.array(x)
    with pytest.raises(ValueError):
        mnm.batch_flatten(x)


def test_batch_flatten_2d():
    x = np.random.randn(5, 3).astype("float32")
    x = mnm.array(x)
    y = mnm.batch_flatten(x)
    assert y.shape == (5, 3)


def test_batch_flatten_3d():
    x = np.random.randn(5, 3, 2).astype("float32")
    x = mnm.array(x)
    y = mnm.batch_flatten(x)
    assert y.shape == (5, 6)


def test_batch_flatten_4d():
    x = np.random.randn(5, 2, 2, 2).astype('float32')
    x = mnm.array(x)
    y = mnm.batch_flatten(x)
    assert y.shape == (5, 8)


if __name__ == "__main__":
    test_batch_flatten_0d()
    test_batch_flatten_1d()
    test_batch_flatten_2d()
    test_batch_flatten_3d()
    test_batch_flatten_4d()
