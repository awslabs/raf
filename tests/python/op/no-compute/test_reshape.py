# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

import raf


@pytest.mark.parametrize("shapes", [((4, 4), (4, 2)), ((5, 3), (5, 5))])
def test_reshape_error(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = raf.array(x)
    with pytest.raises(ValueError):
        raf.reshape(x, reshape)


@pytest.mark.parametrize(
    "shapes", [((4, 4, 4), (4, 2, 8)), ((5, 3, 2), (5, 6)), ((5, 6), (3, 2, 5))]
)
def test_reshape(shapes):
    shape, reshape = shapes
    x = np.random.randn(*shape).astype("float32")
    x = raf.array(x)
    y = raf.reshape(x, reshape)
    assert y.shape == reshape


def test_create_view_with_empty_shape():
    x = np.random.randn(1).astype("float32")
    x = raf.array(x)
    y = raf.squeeze(x)
    assert y.shape == ()


if __name__ == "__main__":
    pytest.main([__file__])
