# pylint: disable=protected-access
import pytest
import tvm
import mnm
from tvm import relay
from mnm.testing import check, randn
from mnm._core.ir_ext import extended_var, ExtendedVar
from mnm._core.module import IRModule
from mnm._ffi.model import RunModel
from mnm.model.trace import _unwrap


def test_ext_constant():
    def expected():
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = mnm._core.value.TensorValue.from_numpy(x_const.numpy())
        x = mnm._ffi.ir._make.Constant(x_value)
        pooled = mnm.ir.op.max_pool2d(x, 3, 1, 0)
        ovar = extended_var("out")
        let = relay.Let(ovar, pooled, ovar)
        return relay.Function([], let)

    func_origin = expected()
    mod_origin = IRModule.from_expr(func_origin)
    out_origin = _unwrap(RunModel(mod_origin, []))

    json = mnm.ir.save_json(func_origin)

    func_loaded = tvm.ir.load_json(json)
    mod_loaded = IRModule.from_expr(func_loaded)
    out_loaded = _unwrap(RunModel(mod_loaded, []))

    assert tvm.ir.structural_equal(func_loaded, func_origin)
    check(out_origin, out_loaded)


def test_tuple_ext_constant():
    def expected():
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = mnm._core.value.TensorValue.from_numpy(x_const.numpy())
        x = mnm._ffi.ir._make.Constant(x_value)
        pooled = mnm.ir.op.max_pool2d(x, 3, 1, 0)
        ovar = extended_var("out")
        let = relay.Let(ovar, pooled, ovar)
        return relay.Function([], let)

    func_origin = expected()
    mod_origin = IRModule.from_expr(func_origin)
    out_origin = _unwrap(RunModel(mod_origin, []))
    m = {"a": 1, "b": 2}

    json = mnm.ir.save_json((func_origin, m))

    loaded = tvm.ir.load_json(json)
    assert len(loaded) == 2

    func_loaded, m_loaded = loaded
    mod_loaded = IRModule.from_expr(func_loaded)
    out_loaded = _unwrap(RunModel(mod_loaded, []))

    assert tvm.ir.structural_equal(func_loaded, func_origin)
    assert len(m_loaded) == 2
    assert m_loaded["a"] == 1
    assert m_loaded["b"] == 2
    check(out_origin, out_loaded)


def test_large_grpah():
    def expected():
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = mnm._core.value.TensorValue.from_numpy(x_const.numpy())
        x = mnm._ffi.ir._make.Constant(x_value)
        size = int(1e5)
        var = [extended_var("var_" + str(i)) for i in range(size)]
        body = var[-1]
        for i in range(size, 1, -1):
            body = relay.Let(var[i - 1], relay.Call(add_op, [var[i - 2], x]), body)
        return relay.Function([var[0]], body)

    func_origin = expected()
    json = mnm.ir.save_json(func_origin)
    tvm.ir.load_json(json)


def test_extended_var():
    # FIXME: serialization does not preserve may_share
    def expected():
        x = extended_var("x")
        y = mnm.ir.op.relu(x)
        ovar = extended_var("out", may_share=x)
        let = relay.Let(ovar, y, ovar)
        return relay.Function([x], let)

    func_origin = expected()

    json = mnm.ir.save_json(func_origin)

    func_loaded = mnm.ir.load_json(json)

    # ensure AsText does not segfault
    _ = mnm.ir.AsText(func_loaded)
    assert tvm.ir.structural_equal(func_loaded, func_origin)
    assert ExtendedVar(func_loaded.body.var).may_share is None


if __name__ == "__main__":
    pytest.main([__file__])
