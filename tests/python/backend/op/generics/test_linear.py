from mnm._core.op import invoke_make_output
from mnm._core.value import TensorValue


def test_1d():
    data = TensorValue.assemble((5, ), "float32", "cpu")
    weight = TensorValue.assemble((10, 5), "float32", "cpu")
    out = invoke_make_output("mnm.op.linear", "", (data, weight))
    assert(out.shape == (10, ))


def test_2d():
    data = TensorValue.assemble((5, 10), "float32", "cpu")
    weight = TensorValue.assemble((20, 10), "float32", "cpu")
    out = invoke_make_output("mnm.op.linear", "", (data, weight))
    assert(out.shape == (5, 20))


def test_3d():
    data = TensorValue.assemble((5, 20, 10), "float32", "cpu")
    weight = TensorValue.assemble((20, 10), "float32", "cpu")
    out = invoke_make_output("mnm.op.linear", "", (data, weight))
    assert(out.shape == (5, 20, 20))


if __name__ == "__main__":
    test_1d()
    test_2d()
    test_3d()
