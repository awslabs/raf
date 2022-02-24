# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import raf


def _shape(shape):
    if shape is None:
        return ()
    return tuple(shape)


@pytest.mark.parametrize("shape", [None, [], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_uniform(shape):
    assert raf.random.uniform(shape=shape).shape == _shape(shape)


@pytest.mark.parametrize("shape", [None, [], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_normal(shape):
    assert raf.random.normal(shape=shape).shape == _shape(shape)


if __name__ == "__main__":
    pytest.main([__file__])
