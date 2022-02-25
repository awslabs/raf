# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import raf


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_xavier_normal(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            raf.random.nn.xavier_normal(shape, gain=1.0)
    else:
        raf.random.nn.xavier_normal(shape, gain=1.0)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_xavier_uniform(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            raf.random.nn.xavier_uniform(shape, gain=1.0)
    else:
        raf.random.nn.xavier_uniform(shape, gain=1.0)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_kaiming_normal(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            raf.random.nn.kaiming_normal(shape)
    else:
        raf.random.nn.kaiming_normal(shape)


@pytest.mark.parametrize("shape", [[], [1], [2, 1], [4, 4, 4, 4, 4]])
def test_kaiming_uniform(shape):
    if len(shape) < 2:
        with pytest.raises(ValueError):
            raf.random.nn.kaiming_uniform(shape)
    else:
        raf.random.nn.kaiming_uniform(shape)


if __name__ == "__main__":
    pytest.main([__file__])
