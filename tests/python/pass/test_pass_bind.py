import pytest
import mnm

from mnm._lib import relay, tvm
from mnm._ffi.pass_ import Substitute
from mnm._core.ir_ext import ExtendedVar, extended_var


def test_basic():
    # Create a symbolic model and run it
    x = relay.var('x')
    y = relay.var('y')
    a = relay.var('a')
    b = relay.var('b')
    expr = mnm.ir.op.add(x, y)
    vmap = {
        x: mnm.ir.op.multiply(a, b),
        y: mnm.ir.op.add(a, b)
    }

    def expected():
        return mnm.ir.op.add(mnm.ir.op.multiply(a, b), mnm.ir.op.add(a, b))

    expr_after = Substitute(expr, vmap)
    expr_expected = expected()
    tvm.ir.structural_equal(expr_after, expr_expected)


def test_extended_var():
    # pylint: disable=invalid-name
    p = relay.var('p')
    r = relay.var('r')
    m = relay.var('m')
    n = relay.var('n')
    x = extended_var('x', may_share=p)
    y = extended_var('y', may_share=p)
    expr = relay.Let(x, mnm.ir.op.add(p, r), relay.Let(y, mnm.ir.op.add(x, r), y))
    vmap = {
        p: m,
        r: n
    }

    def expected():
        return relay.Let(x, mnm.ir.op.add(m, n), relay.Let(y, mnm.ir.op.add(x, n), y))

    expr_after = Substitute(expr, vmap)
    expr_expected = expected()
    tvm.ir.structural_equal(expr_after, expr_expected)
    assert ExtendedVar(expr_after.var).may_share == m
    assert ExtendedVar(expr_after.body.var).may_share == m


if __name__ == "__main__":
    pytest.main([__file__])
