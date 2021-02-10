# pylint: disable=invalid-name,protected-access
import numpy as np
import pytest
import mnm
import tvm

from mnm.testing import get_device_list, randn, check
from tvm import relay


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3, 3],
    [4, 4]
])
def test_add_to(shape, device):
    class Add(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm.add(x, x)
    model = Add()
    m_x, _ = randn(shape, device=device, requires_grad=True)
    m_y = model(m_x)
    m_dy, n_dy = randn(shape, device=device)
    m_y.backward(m_dy)
    m_dx = m_x.grad
    n_dx = 2 * n_dy
    check(m_dx, n_dx)


@pytest.mark.parametrize("device", get_device_list())
@pytest.mark.parametrize("shape", [
    [3,],
    [4,]
])
def test_no_grad1(shape, device):
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y, z):  # pylint: disable=no-self-use
            indices = mnm.add(y, z)
            indices = mnm.subtract(indices, z)
            indices = mnm.add(indices, z)
            return mnm.take(x, indices, axis=0)

    model = Model()
    m_x, n_x = randn(shape, device=device, requires_grad=True)
    m_y = mnm.array([1,], dtype="int64", device=device)
    m_z = mnm.array([1,], dtype="int64", device=device)
    m_out = model(m_x, m_y, m_z)  # m_out = m_x[2]
    m_dout, n_dout = randn([1,], device=device)
    m_out.backward(m_dout)
    m_dx = m_x.grad
    n_dx = np.zeros_like(n_x)
    n_dx[2] = n_dout[0]
    check(m_dx, n_dx)


@pytest.mark.parametrize("device", get_device_list())
def test_no_grad2(device):
    matmul_op = mnm._ffi.op.GetOp("mnm.op.matmul")
    matmul_nt_op = mnm._ffi.op.GetOp("mnm.op.matmul_nt")
    shape = [3, 2]
    dtype = "float32"

    def expected():
        x = relay.var("x", shape=shape)
        y = relay.var("y", shape=shape)
        a1 = relay.var("a1")
        closure = relay.var("closure")
        dy = relay.var("dy")
        x1 = relay.var("x")
        x2 = relay.var("x")
        ret = relay.var("ret")
        inner_let2 = relay.Let(
            x2, relay.Tuple([x1, mnm._ffi.ir._make.Constant(mnm._core.value.NoGradValue())]), x2)
        inner_let1 = relay.Let(x1, relay.Call(matmul_op, [dy, y]), inner_let2)
        let3 = relay.Let(ret, relay.Tuple([a1, closure]), ret)
        let2 = relay.Let(closure, relay.Function([dy], body=inner_let1), let3)
        let1 = relay.Let(a1, relay.Call(matmul_nt_op, [x, y]), let2)
        return relay.Function([x, y], let1)

    class Model(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, y):    # pylint: disable=no-self-use
            return mnm.matmul_nt(x, y)

    model = Model()
    # forward
    m_x, _ = randn(shape, dtype=dtype, device=device)
    m_y, _ = randn(shape, dtype=dtype, device=device)
    m_x.requires_grad = True
    m_y.requires_grad = False

    m_record = model._internal(m_x, m_y)
    # backward
    m_func = mnm._ffi.pass_.AutoDiff(m_record.func, m_record.requires_grads)
    assert tvm.ir.structural_equal(m_func, expected())


if __name__ == "__main__":
    pytest.main([__file__])
