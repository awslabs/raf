# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import mnm


def test_mnm_array_cpu():
    array = mnm.array([1, 2, 3], dtype="int8", device="cpu")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


@pytest.mark.skipif(not mnm.build.with_cuda(), reason="CUDA is not enabled")
def test_mnm_array_cuda():
    array = mnm.array([1, 2, 3], dtype="int8", device="cuda")
    array = array.numpy()
    assert np.all(array == [1, 2, 3])


if __name__ == "__main__":
    test_mnm_array_cpu()
    test_mnm_array_cuda()
