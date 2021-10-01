# pylint: disable=protected-access, no-self-use
import pytest
import mnm
from mnm._ffi.pass_ import SimplifyExpr, ToGraphNormalForm, ToBasicBlockNormalForm
from mnm.ir import MNMSequential
from mnm.testing import randn

def simplify(mod):
    seq = MNMSequential([ToGraphNormalForm(), ToBasicBlockNormalForm(), SimplifyExpr()])
    return seq(mod)


@pytest.mark.parametrize("op", ["zeros_like", "ones_like"])
def test_unary_like(op):
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            return getattr(mnm._op.sym, op)(x)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")

    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "like" not in text, text


@pytest.mark.parametrize("params", [("cast_like", (10, 1), "float16"),
                                    ("broadcast_to_like", (10, 10), "float32")])
def test_binary_like(params):
    op, shape_like, dtype_like = params
    device = "cpu"
    shape = (10, 1)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            return getattr(mnm._op.sym, op)(x, y)

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    m_y, _ = randn(shape_like, device=device, dtype=dtype_like)

    mod = model._internal(m_x, m_y).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "like" not in text, text


def test_cast():
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.cast(x, "float16")
            y = mnm.cast(y, "float32")
            y = mnm.cast(y, "float16")
            y = mnm.cast(y, "float32")
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "mnm.op.cast" not in text, text


def test_reshape():
    device = "cpu"
    shape = (10, 5)

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            y = mnm.reshape(x, (shape[0] * shape[1],))
            y = mnm.reshape(y, (shape[1], shape[0]))
            y = mnm.reshape(y, shape)
            return y

    model = Model()
    m_x, _ = randn(shape, device=device, dtype="float32")
    mod = model._internal(m_x).mod
    mod = simplify(mod)
    text = mnm.ir.AsText(mod["main"])
    assert "mnm.op.reshape" not in text, text


if __name__ == "__main__":
    pytest.main([__file__])
