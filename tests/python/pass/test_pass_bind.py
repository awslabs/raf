# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import raf

from raf._lib import relay, tvm
from raf._ffi.pass_ import Substitute
from raf._core.ir_ext import ExtendedVar, extended_var


def test_basic():
    # Create a symbolic model and run it
    x = relay.var("x")
    y = relay.var("y")
    a = relay.var("a")
    b = relay.var("b")
    expr = raf.ir.op.add(x, y)
    vmap = {x: raf.ir.op.multiply(a, b), y: raf.ir.op.add(a, b)}

    def expected():
        return raf.ir.op.add(raf.ir.op.multiply(a, b), raf.ir.op.add(a, b))

    expr_after = Substitute(expr, vmap)
    expr_expected = expected()
    tvm.ir.structural_equal(expr_after, expr_expected)


def test_extended_var():
    # pylint: disable=invalid-name
    p = relay.var("p")
    r = relay.var("r")
    m = relay.var("m")
    n = relay.var("n")
    x = extended_var("x", may_share=p)
    y = extended_var("y", may_share=p)
    expr = relay.Let(x, raf.ir.op.add(p, r), relay.Let(y, raf.ir.op.add(x, r), y))
    vmap = {p: m, r: n}

    def expected():
        return relay.Let(x, raf.ir.op.add(m, n), relay.Let(y, raf.ir.op.add(x, n), y))

    expr_after = Substitute(expr, vmap)
    expr_expected = expected()
    tvm.ir.structural_equal(expr_after, expr_expected)
    assert ExtendedVar(expr_after.var).may_share == m
    assert ExtendedVar(expr_after.body.var).may_share == m


if __name__ == "__main__":
    pytest.main([__file__])
