import functools
import operator
import pytest
import numpy as np

import mnm
from mnm.testing import run_vm_model, check, get_device_list


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
@pytest.mark.parametrize("device", get_device_list())
def test_batch_flatten(shape, device):
    class Model(mnm.model.Model):
        # pylint: disable=no-self-use
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return mnm.batch_flatten(x)

    x = np.random.randn(*shape).astype("float32")
    x = mnm.array(x)
    # imperative
    y_i = mnm.batch_flatten(x)
    expected = (5, functools.reduce(operator.mul, list(x.shape)[1:]))
    assert y_i.shape == expected
    dy = mnm.reshape(y_i, mnm.shape(x))
    assert dy.shape == x.shape
    assert (x.asnumpy() == dy.asnumpy()).all()
    # traced
    model = Model()
    y_t = run_vm_model(model, device, [x])
    check(y_t, y_i)


if __name__ == "__main__":
    pytest.main([__file__])
