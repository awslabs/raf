from mnm._ffi.op import invoke_make_output
from mnm._ffi.value import TensorValue


def test_1d():
    data = TensorValue.assemble((5, ), "float32", "cpu")
    try:
        invoke_make_output("mnm.op.batch_flatten", "", data)
    except ValueError:
        pass
    else:
        raise AssertionError("Should raise exception")


def test_2d():
    data = TensorValue.assemble((5, 3), "float32", "cpu")
    out = invoke_make_output("mnm.op.batch_flatten", "", data)
    assert out.shape == (5, 3)


def test_3d():
    data = TensorValue.assemble((5, 3, 2), "float32", "cpu")
    out = invoke_make_output("mnm.op.batch_flatten", "", data)
    assert out.shape == (5, 6)


if __name__ == "__main__":
    test_1d()
    test_2d()
    test_3d()
