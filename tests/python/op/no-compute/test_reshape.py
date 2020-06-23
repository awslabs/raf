import pytest
import numpy as np

import mnm


@pytest.mark.parametrize("shapes", [((4, 4), (4, 2)), ((5, 3), (5, 5))])
def test_reshape_error(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    with pytest.raises(ValueError):
        mnm.reshape(x, reshape)


@pytest.mark.parametrize("shapes", [
    ((4, 4, 4), (4, 2, 8)),
    ((5, 3, 2), (5, 6)),
    ((5, 6), (3, 2, 5))
])
def test_reshape(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    y = mnm.reshape(x, reshape)
    assert y.shape == reshape


if __name__ == "__main__":
    pytest.main([__file__])
