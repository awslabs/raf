# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import operator
import pytest
import numpy as np

import raf
from raf.testing import run_vm_model, check, get_testable_devices


@pytest.mark.parametrize("shape", [(1, ()), (5, (5,))])
def test_batch_flatten_error(shape):
    shape, reshape = shape
    x = np.random.randn(shape).astype("float32").reshape(reshape)
    x = raf.array(x)
    assert x.shape == reshape
    with pytest.raises(ValueError):
        raf.batch_flatten(x)


@pytest.mark.parametrize("shape", [[5, 3], [5, 2, 2, 2]])
@pytest.mark.parametrize("device", get_testable_devices())
def test_batch_flatten(shape, device):
    class Model(raf.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf.batch_flatten(x)

    x = np.random.randn(*shape).astype("float32")
    x = raf.array(x)
    # imperative
    y_i = raf.batch_flatten(x)
    expected = (5, functools.reduce(operator.mul, list(x.shape)[1:]))
    assert y_i.shape == expected
    dy = raf.reshape(y_i, raf.shape(x))
    assert dy.shape == x.shape
    assert (x.numpy() == dy.numpy()).all()
    # traced
    model = Model()
    y_t = run_vm_model(model, device, [x])
    check(y_t, y_i)


if __name__ == "__main__":
    pytest.main([__file__])
