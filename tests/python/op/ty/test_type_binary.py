# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
from raf._op import sym
from raf._ffi.pass_ import AutoDiff, InferType
from raf.testing import check_type, run_infer_type, randn
from tvm.relay import TensorType, FuncType, TupleType


@pytest.mark.parametrize(
    "op",
    [
        sym.add,
        sym.subtract,
        sym.multiply,
        sym.divide,
        sym.floor_divide,
        sym.mod,
        sym.maximum,
        sym.minimum,
        sym.right_shift,
        sym.left_shift,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [(10, 4), (5, 10, 1), (5, 10, 4)],
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "uint32", "int32", "int64", "uint8", "int16"])
def test_binary(op, shape, dtype):
    # pylint: disable=too-many-locals
    class Binary(raf.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @raf.model.trace
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    s_a, s_b, s_c = shape
    t_a = TensorType(s_a, dtype=dtype)
    t_b = TensorType(s_b, dtype=dtype)
    t_c = TensorType(s_c, dtype=dtype)
    # check forward
    m_a, _ = randn(s_a, dtype=dtype)
    m_b, _ = randn(s_b, dtype=dtype)
    m_a.requires_grad = True
    m_b.requires_grad = True
    record = model._internal(m_a, m_b)
    m_mod = record.mod
    m_mod = InferType()(m_mod)
    desired_type = FuncType([t_a, t_b], t_c)
    check_type(m_mod["main"], desired_type)
    # check backward
    # TODO(yzhliu): some operators are missing gradient registries.
    if op not in (sym.mod, sym.maximum, sym.minimum, sym.subtract):
        bwd_mod = AutoDiff(record.requires_grads)(m_mod)
        bwd_mod = InferType()(bwd_mod)
        bwd_func = bwd_mod["main"]
        bwd_ty = FuncType([t_c], TupleType([t_a, t_b]))
        desired_type = FuncType([t_a, t_b], TupleType([t_c, bwd_ty]))
        check_type(bwd_func, desired_type)


@pytest.mark.parametrize(
    "op",
    [
        sym.less,
        sym.greater,
        sym.less_equal,
        sym.greater_equal,
        sym.equal,
        sym.not_equal,
        sym.logical_and,
    ],
)
@pytest.mark.parametrize("shape", [[(10, 4), (5, 10, 1), (5, 10, 4)]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_logiacal(op, shape, dtype):
    # pylint: disable=too-many-locals
    class Binary(raf.Model):
        def build(self):
            pass

        # pylint: disable=no-self-use
        @raf.model.trace
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    s_a, s_b, s_c = shape
    t_a = TensorType(s_a, dtype=dtype)
    t_b = TensorType(s_b, dtype=dtype)
    t_c = TensorType(s_c, dtype="bool")
    # check forward
    m_a, _ = randn(s_a, dtype=dtype)
    m_b, _ = randn(s_b, dtype=dtype)
    m_func = model._internal(m_a, m_b).mod["main"]
    m_func = run_infer_type(m_func)
    desired_type = FuncType([t_a, t_b], t_c)
    check_type(m_func, desired_type)
    # TODO(@hzfan): check backward. missing gradient registries.


if __name__ == "__main__":
    pytest.main([__file__])
