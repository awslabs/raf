# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, protected-access
import pytest
import tvm
import raf

from raf._core.ndarray import Symbol
from raf._ffi.binding import BindSymbol
from raf._ffi.pass_ import ExtractBinding, ToGraphNormalForm


def test_simple_op():
    v_a = BindSymbol(None, "a", None)
    s_a = Symbol.from_expr(v_a)
    s_b = raf._op.sym.sum(s_a)
    c1 = ExtractBinding(s_b._Symbol__handle, []).value
    c2 = raf.ir.op.sum(v_a)
    assert tvm.ir.structural_equal(c1, c2)


def test_tuple1():
    v_a = BindSymbol(None, "a", None)
    v_b = BindSymbol(None, "b", None)
    s_a = Symbol.from_expr(v_a)
    s_b = Symbol.from_expr(v_b)
    s_c = raf._op.sym.stack([s_a, s_b])
    m1 = raf.ir.IRModule.from_expr(ExtractBinding(s_c._Symbol__handle, []))
    m1 = ToGraphNormalForm()(m1)
    m2 = raf.ir.op.stack([v_a, v_b])
    m2 = raf.ir.IRModule.from_expr(m2)
    assert tvm.ir.structural_equal(m1, m2)


def test_tuple2():
    v_a = BindSymbol(None, "a", None)
    v_b = BindSymbol(None, "b", None)
    s_a = Symbol.from_expr(v_a)
    s_b = Symbol.from_expr(v_b)
    s_c = raf._op.sym.split(s_a, [2, s_b])
    m1 = raf.ir.IRModule.from_expr(ExtractBinding(s_c._Symbol__handle, []))
    m1 = ToGraphNormalForm()(m1)
    m2 = raf.ir.op.split(v_a, [2, v_b])
    m2 = raf.ir.IRModule.from_expr(m2)
    assert tvm.ir.structural_equal(m1["main"].body, m2["main"].body)


if __name__ == "__main__":
    pytest.main([__file__])
