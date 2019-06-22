from mnm import cpu
from mnm.value import TensorValue
from utils import invoke_make_output


def test_0d():
    data = TensorValue.assemble((), "float32", cpu(0))
    out = invoke_make_output("mnm.op.relu", "", data)
    assert data.shape == out.shape


def test_1d():
    data = TensorValue.assemble((5, 3), "float32", cpu(0))
    out = invoke_make_output("mnm.op.relu", "", data)
    assert data.shape == out.shape


def test_2d():
    data = TensorValue.assemble((5, 3, 2), "float32", cpu(0))
    out = invoke_make_output("mnm.op.relu", "", data)
    assert data.shape == out.shape


if __name__ == "__main__":
    test_0d()
    test_1d()
    test_2d()
