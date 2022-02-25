# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import tvm
import raf
from tvm import relay
from raf.testing import check, randn
from raf._core.ir_ext import extended_var, ExtendedVar
from raf._core.module import IRModule
from raf._ffi.model import RunModel
from raf.model.trace import _unwrap


def test_ext_constant():
    def expected():
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = raf._core.value.TensorValue.from_numpy(x_const.numpy())
        x = raf._ffi.ir._make.Constant(x_value)
        pooled = raf.ir.op.max_pool2d(x, 3, 1, 0)
        ovar = extended_var("out")
        let = relay.Let(ovar, pooled, ovar)
        return relay.Function([], let)

    func_origin = expected()
    mod_origin = IRModule.from_expr(func_origin)
    out_origin = _unwrap(RunModel(mod_origin, []))

    json = raf.ir.save_json(func_origin)

    func_loaded = tvm.ir.load_json(json)
    mod_loaded = IRModule.from_expr(func_loaded)
    out_loaded = _unwrap(RunModel(mod_loaded, []))

    assert tvm.ir.structural_equal(func_loaded, func_origin)
    check(out_origin, out_loaded)


def test_tuple_ext_constant():
    def expected():
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = raf._core.value.TensorValue.from_numpy(x_const.numpy())
        x = raf._ffi.ir._make.Constant(x_value)
        pooled = raf.ir.op.max_pool2d(x, 3, 1, 0)
        ovar = extended_var("out")
        let = relay.Let(ovar, pooled, ovar)
        return relay.Function([], let)

    func_origin = expected()
    mod_origin = IRModule.from_expr(func_origin)
    out_origin = _unwrap(RunModel(mod_origin, []))
    m = {"a": 1, "b": 2}

    json = raf.ir.save_json((func_origin, m))

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
        add_op = raf._ffi.op.GetOp("raf.op.add")
        x_const, _ = randn((1, 3, 8, 8), device="cpu")
        x_value = raf._core.value.TensorValue.from_numpy(x_const.numpy())
        x = raf._ffi.ir._make.Constant(x_value)
        size = int(1e5)
        var = [extended_var("var_" + str(i)) for i in range(size)]
        body = var[-1]
        for i in range(size, 1, -1):
            body = relay.Let(var[i - 1], relay.Call(add_op, [var[i - 2], x]), body)
        return relay.Function([var[0]], body)

    func_origin = expected()
    json = raf.ir.save_json(func_origin)
    tvm.ir.load_json(json)


def test_extended_var():
    # FIXME: serialization does not preserve may_share
    def expected():
        x = extended_var("x")
        y = raf.ir.op.relu(x)
        ovar = extended_var("out", may_share=x)
        let = relay.Let(ovar, y, ovar)
        return relay.Function([x], let)

    func_origin = expected()

    json = raf.ir.save_json(func_origin)

    func_loaded = raf.ir.load_json(json)

    # ensure AsText does not segfault
    _ = raf.ir.AsText(func_loaded)
    assert tvm.ir.structural_equal(func_loaded, func_origin)
    assert ExtendedVar(func_loaded.body.var).may_share is None


if __name__ == "__main__":
    pytest.main([__file__])
