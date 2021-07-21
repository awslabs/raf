import pytest
import numpy as np

import mnm


@pytest.mark.parametrize("shape", [
    [5, 3],
    [5, 3, 2],
    [5, 2, 2, 2],
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("num_newaxis", [1, 2, 3])
def test_batch_flatten(shape, axis, num_newaxis):
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    y = mnm.expand_dims(x, axis=axis, num_newaxis=num_newaxis)
    if axis < 0:
        axis = len(shape) + axis + 1
    expected = shape[:axis] + [1] * num_newaxis + shape[axis:]
    assert list(y.shape) == expected
    dy = mnm.reshape(y, mnm.shape(x))
    assert dy.shape == x.shape
    assert (x.numpy() == dy.numpy()).all()


if __name__ == "__main__":
    pytest.main([__file__])
