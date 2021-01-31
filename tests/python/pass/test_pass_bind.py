import pytest
from mnm._lib import relay, tvm
from mnm._ffi.pass_ import Substitute
from mnm._core.ir_ext import ExtendedVar, extended_var


def test_basic():
    # Create a symbolic model and run it
    x = relay.var('x')
    y = relay.var('y')
    a = relay.var('a')
    b = relay.var('b')
    add = relay.op.get("mnm.op.add")
    multiply = relay.op.get("mnm.op.multiply")
    expr = relay.Call(add, [x, y])
    vmap = {
        x: relay.Call(multiply, [a, b]),
        y: relay.Call(add, [a, b])
    }

    def expected():
        return relay.Call(add, [relay.Call(multiply, [a, b]), relay.Call(add, [a, b])])

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
    add = relay.op.get("mnm.op.add")
    expr = relay.Let(x, relay.Call(add, [p, r]), relay.Let(y, relay.Call(add, [x, r]), y))
    vmap = {
        p: m,
        r: n
    }

    def expected():
        return relay.Let(x, relay.Call(add, [m, n]), relay.Let(y, relay.Call(add, [x, n]), y))

    expr_after = Substitute(expr, vmap)
    expr_expected = expected()
    tvm.ir.structural_equal(expr_after, expr_expected)
    assert ExtendedVar(expr_after.var).may_share == m
    assert ExtendedVar(expr_after.body.var).may_share == m


if __name__ == "__main__":
    pytest.main([__file__])
