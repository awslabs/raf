# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import raf


def test_raf_array_cpu():
    array = raf.array([1, 2, 3], dtype="int8", device="cpu")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_raf_array_cuda():
    array = raf.array([1, 2, 3], dtype="int8", device="cuda")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


if __name__ == "__main__":
    test_raf_array_cpu()
    test_raf_array_cuda()
