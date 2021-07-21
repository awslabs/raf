# pylint: disable=no-self-use
# pylint: disable=wrong-import-order
import numpy as np
import random
import pytest
import mnm
import mxnet as mx
from mnm.testing import get_device_list, randn, check, run_vm_model


class TestModel(mnm.Model):
    def build(self, op, **kwargs):
        self.op = op  # pylint: disable=attribute-defined-outside-init
        self.attrs = kwargs  # pylint: disable=attribute-defined-outside-init

    @mnm.model.trace
    def forward(self, *args):
        return self.op(*args, **self.attrs)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_argsort(device, shape, axis, dtype):
    m_x, n_x = randn(shape, device=device)
    model = TestModel(mnm._op.sym.argsort, axis=axis, dtype=dtype)  # pylint: disable=protected-access
    m_out = model(m_x)
    v_out = run_vm_model(model, device, [m_x])
    np_out = np.argsort(n_x, axis).astype(dtype)
    check(m_out, np_out)
    check(v_out, np_out)


# pylint: disable=too-many-locals
# pylint: disable=no-member
# pylint: disable=consider-using-in
@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    (2, 3, 4),
    (1, 4, 6),
    (3, 5, 6),
])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_sort(device, shape, axis, dtype):
    m_x, n_x = randn(shape, device=device, dtype=dtype)
    m_x.requires_grad = True
    model = TestModel(mnm._op.sym.sort, axis=axis)  # pylint: disable=protected-access
    m_out = model(m_x)
    v_out = run_vm_model(model, device, [m_x])
    np_out = np.sort(n_x, axis)
    check(m_out, np_out)
    check(v_out, np_out)
    if dtype == "float32" or dtype == "float64":
        m_dy, n_dy = randn(m_out.shape, device=device, dtype=dtype)
        m_out.backward(m_dy)

        # ground truth
        mx_x = mx.nd.array(n_x)
        mx_x.attach_grad()
        mx_dy = mx.nd.array(n_dy)
        with mx.autograd.record():
            mx_y = mx.nd.sort(mx_x, axis)
            mx_y.backward(mx_dy)
        check(mx_x.grad, m_x.grad)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("axis", [0, 2, -1])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("ret_type", ["values", "both", "indices"])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("shape", [(5, 3, 3), (5, 5, 5, 5, 5, 5, 5), (224, 224, 3)])
def test_topk(shape, k, axis, ret_type, is_ascend, dtype, device):
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=not-callable
    # pylint: disable=no-member
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=no-self-use
    size = 1
    for i in shape:
        size *= i
    x = np.arange(size)
    random.shuffle(x)
    x = x.reshape(shape)
    m_x = mnm.array(x, dtype=dtype, device=device)
    n_x = mx.nd.array(x, dtype=dtype)
    # uncomment once topk_dx is merged.
    # m_x.requires_grad = True

    model = TestModel(mnm._op.sym.topk, k=k, axis=axis, ret_type=ret_type,  # pylint: disable=protected-access
                      is_ascend=is_ascend, dtype=dtype)
    m_y = model(m_x)
    v_y = run_vm_model(model, device, [m_x])
    # check forward
    if ret_type == "values":
        n_y = mx.nd.topk(n_x, k=k, axis=axis, ret_typ="value", is_ascend=is_ascend, dtype=dtype)
        check(m_y, n_y)
        check(v_y, n_y)
    else:
        n_y = mx.nd.topk(n_x, k=k, axis=axis, ret_typ=ret_type, is_ascend=is_ascend, dtype=dtype)
        if ret_type == "both":
            check(m_y[0], n_y[0])
            check(v_y[0], n_y[0])
            check(m_y[1], n_y[1])
            check(v_y[1], n_y[1])
        else:
            check(m_y, n_y)
            check(v_y, n_y)


if __name__ == "__main__":
    pytest.main([__file__])
