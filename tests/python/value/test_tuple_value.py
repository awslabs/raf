from mnm._ffi.value import TensorValue, TupleValue


def test_nested():
    d_0 = TensorValue.assemble((), "float32", "cpu")
    d_1 = TensorValue.assemble((3, ), "float64", "cpu")
    d_2 = TensorValue.assemble((3, 2), "float16", "cpu")
    d_3 = TensorValue.assemble((3, 2, 6), "float32", "cpu")
    v_0 = TupleValue([d_0, d_1, d_2, d_3])
    v_1 = TupleValue([v_0, v_0])
    v_2 = TupleValue([])

    assert len(v_0) == 4
    assert len(v_1) == 2
    assert len(v_1[0]) == 4
    assert len(v_1[1]) == 4
    assert len(v_2) == 0
    assert not v_2

    assert v_0[0].same_as(d_0)
    assert v_0[1].same_as(d_1)
    assert v_0[2].same_as(d_2)
    assert v_0[3].same_as(d_3)
    assert v_1[0][0].same_as(d_0)
    assert v_1[0][1].same_as(d_1)
    assert v_1[0][2].same_as(d_2)
    assert v_1[0][3].same_as(d_3)
    assert v_1[1][0].same_as(d_0)
    assert v_1[1][1].same_as(d_1)
    assert v_1[1][2].same_as(d_2)
    assert v_1[1][3].same_as(d_3)


if __name__ == "__main__":
    test_nested()
