# pylint: disable=protected-access, too-many-locals
import pytest
import mnm
from mnm.frontend import FrameworkModel
from mnm.testing import randn, check
from mnm._ffi.pass_ import FromRelay
from mnm._lib import tvm as _tvm
from mnm._lib import relay as _relay


@pytest.mark.parametrize("shape", [(2, 2)])
def test_mnm_add(shape):
    # meta ir
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, y):  # pylint: disable=no-self-use
            return mnm.add(x, y)

    model = TestModel()
    m_x, _ = randn(shape)
    m_y, _ = randn(shape)
    m_func = model._internal(m_x, m_y).func
    m_z = model(m_x, m_y)
    # relay ir
    r_x = _relay.var("x", shape=shape)
    r_y = _relay.var("y", shape=shape)
    r_func = _relay.Function(params=[r_x, r_y], body=r_x+r_y)
    # check
    new_func = FromRelay(r_func)
    assert _tvm.ir.structural_equal(m_func, new_func)
    new_model = FrameworkModel(new_func, new_func, {}, {})
    check(m_z, new_model(m_x, m_y))


@pytest.mark.parametrize("xshape", [(8, 3, 32, 32)])
@pytest.mark.parametrize("wshape", [(16, 3, 3, 3)])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
def test_mnm_conv2d(xshape, wshape, stride, dilation, padding):
    # meta ir
    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x, w):  # pylint: disable=no-self-use
            return mnm.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=1)

    model = TestModel()
    m_x, _ = randn(xshape)
    m_w, _ = randn(wshape)
    m_func = model._internal(m_x, m_w).func
    m_y = model(m_x, m_w)
    # relay ir
    r_x = _relay.var("x", shape=xshape)
    r_w = _relay.var("w", shape=wshape)
    r_c = _relay.nn.conv2d(r_x, r_w, strides=stride, dilation=dilation, padding=padding)
    r_func = _relay.Function(params=[r_x, r_w], body=r_c)
    # check
    new_func = FromRelay(r_func)
    assert _tvm.ir.structural_equal(m_func, new_func)
    new_model = FrameworkModel(new_func, new_func, {}, {})
    check(m_y, new_model(m_x, m_w))


@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize(
    "funcs",
    [
        [mnm._op.sym.max_pool2d, _relay.nn.max_pool2d],
        [mnm._op.sym.avg_pool2d, _relay.nn.avg_pool2d],
    ])
def test_mnm_pool2d(kernel, stride, padding, funcs):
    mnm_fwd, relay_fwd = funcs
    if padding > kernel // 2:
        return

    class TestModel(mnm.Model):
        def build(self):
            pass
        @mnm.model.trace
        def forward(self, x):  # pylint: disable=no-self-use
            return mnm_fwd(x, kernel=kernel, stride=stride, padding=padding)

    model = TestModel()
    m_x, _ = randn([8, 3, 32, 32])
    m_func = model._internal(m_x).func
    m_y = model(m_x)
    # relay ir
    r_x = _relay.var("x", shape=[8, 3, 32, 32])
    r_c = relay_fwd(r_x, kernel, stride, padding)
    r_func = _relay.Function(params=[r_x], body=r_c)
    # check
    new_func = FromRelay(r_func)
    print(m_func)
    print(new_func)
    assert _tvm.ir.structural_equal(m_func, new_func)
    new_model = FrameworkModel(new_func, new_func, {}, {})
    check(m_y, new_model(m_x))


if __name__ == "__main__":
    pytest.main([__file__])
