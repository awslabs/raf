# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,no-self-use
import pytest
import mnm
from mnm.testing import run_infer_type, randn
from mnm.model import Conv2d
import tvm


def test_simple():
    konst, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.c)
            y = mnm.relu(y)
            y = mnm.log(y)
            return y

    model = Model()
    m_x, _ = randn((10, 20), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm(mod)
    mod = mnm._ffi.pass_.ToANormalForm(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_conv2d():
    rand, _ = randn((1,), device="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
            self.conv2 = Conv2d(16, 16, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)

        @mnm.model.trace
        def forward(self, x):
            x = mnm.add(x, self.c)
            y = self.conv1(x)
            # this is the next dominator.
            y1 = mnm.add(y, self.c)
            y = mnm.add(y, y1)
            # second path
            z2 = self.conv2(y)
            z3 = self.conv3(y)
            # add can only be fused to z1
            z = mnm.add(z2, z3)
            return z

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), device="cpu")
    mod = model._internal(m_x).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm(mod)
    mod = mnm._ffi.pass_.ToANormalForm(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_tuple():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            z = mnm.add(x, y)
            zz = mnm.split(z, 2)
            return zz[0]

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm(mod)
    mod = mnm._ffi.pass_.ToANormalForm(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


def test_diamond():
    konst, _ = randn((1,))
    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x, y):
            z1 = mnm.add(x, y)
            z2 = mnm.multiply(x, self.c)
            return mnm.relu(mnm.add(z1, z2))

    model = Model()
    m_x, _ = randn((10, 20))
    m_y, _ = randn((10, 1))
    mod = model._internal(m_x, m_y).mod
    func_before = run_infer_type(mod)["main"]
    mod = mnm._ffi.pass_.ToGraphNormalForm(mod)
    mod = mnm._ffi.pass_.ToANormalForm(mod)
    func_after = run_infer_type(mod)["main"]
    assert tvm.ir.structural_equal(func_after, func_before)


if __name__ == "__main__":
    pytest.main([__file__])
