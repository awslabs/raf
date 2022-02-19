# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, protected-access
import pytest
import tvm
import mnm

from mnm._core.ndarray import Symbol
from mnm._ffi.binding import BindSymbol
from mnm._ffi.pass_ import ExtractBinding, ToGraphNormalForm


def test_simple_op():
    v_a = BindSymbol(None, "a", None)
    s_a = Symbol.from_expr(v_a)
    s_b = mnm._op.sym.sum(s_a)
    c1 = ExtractBinding(s_b._Symbol__handle, []).value
    c2 = mnm.ir.op.sum(v_a)
    assert tvm.ir.structural_equal(c1, c2)


def test_tuple1():
    v_a = BindSymbol(None, "a", None)
    v_b = BindSymbol(None, "b", None)
    s_a = Symbol.from_expr(v_a)
    s_b = Symbol.from_expr(v_b)
    s_c = mnm._op.sym.stack([s_a, s_b])
    m1 = mnm.ir.IRModule.from_expr(ExtractBinding(s_c._Symbol__handle, []))
    m1 = ToGraphNormalForm()(m1)
    m2 = mnm.ir.op.stack([v_a, v_b])
    m2 = mnm.ir.IRModule.from_expr(m2)
    assert tvm.ir.structural_equal(m1, m2)


def test_tuple2():
    v_a = BindSymbol(None, "a", None)
    v_b = BindSymbol(None, "b", None)
    s_a = Symbol.from_expr(v_a)
    s_b = Symbol.from_expr(v_b)
    s_c = mnm._op.sym.split(s_a, [2, s_b])
    m1 = mnm.ir.IRModule.from_expr(ExtractBinding(s_c._Symbol__handle, []))
    m1 = ToGraphNormalForm()(m1)
    m2 = mnm.ir.op.split(v_a, [2, v_b])
    m2 = mnm.ir.IRModule.from_expr(m2)
    assert tvm.ir.structural_equal(m1["main"].body, m2["main"].body)


if __name__ == "__main__":
    pytest.main([__file__])
