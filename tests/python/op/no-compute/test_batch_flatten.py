import functools
import operator
import pytest
import numpy as np

import mnm


@pytest.mark.parametrize("shape", [(1, ()), (5, (5,))])
def test_batch_flatten_error(shape):
    shape, reshape = shape
    x = np.random.randn(shape).astype("float32").reshape(reshape)
    x = mnm.array(x)
    assert x.shape == reshape
    with pytest.raises(ValueError):
        mnm.batch_flatten(x)


@pytest.mark.parametrize("shape", [
    [5, 3],
    [5, 3, 2],
    [5, 2, 2, 2]
])
def test_batch_flatten(shape):
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    y = mnm.batch_flatten(x)
    expcected = (5, functools.reduce(operator.mul, list(x.shape)[1:]))
    assert y.shape == expcected
    dy = mnm.batch_flatten_dx(x, y, y)
    assert dy.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
