from mnm import cpu
from mnm.value import TensorValue


def test_assemble_0d():
    data = TensorValue.assemble((), "float32", cpu(0))
    assert data.ndim == 0
    assert data.shape == ()
    assert data.strides == ()


def test_assemble_1d():
    data = TensorValue.assemble((3, ), "float32", cpu(0))
    assert data.ndim == 1
    assert data.shape == (3, )
    assert data.strides == (1, )


def test_assemble_2d():
    data = TensorValue.assemble((3, 2), "float32", cpu(0))
    assert data.ndim == 2
    assert data.shape == (3, 2)
    assert data.strides == (2, 1)


def test_assemble_3d():
    data = TensorValue.assemble((3, 2, 6), "float32", cpu(0))
    assert data.ndim == 3
    assert data.shape == (3, 2, 6)
    assert data.strides == (12, 6, 1)


def test_assemble_null_1d():
    data = TensorValue.assemble((0, ), "float32", cpu(0))
    assert data.ndim == 1
    assert data.shape == (0, )


def test_assemble_null_3d():
    data = TensorValue.assemble((3, 0, 2), "float32", cpu(0))
    assert data.ndim == 3
    assert data.shape == (3, 0, 2)


if __name__ == "__main__":
    test_assemble_0d()
    test_assemble_1d()
    test_assemble_2d()
    test_assemble_3d()
    test_assemble_null_1d()
    test_assemble_null_3d()
