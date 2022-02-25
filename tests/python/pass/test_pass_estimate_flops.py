# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import raf
import tvm
from tvm import relay

from raf._core.device import Device
from raf._ffi.pass_ import EstimateGFLOPS
from raf.ir import ScopeBuilder
from raf.testing import run_infer_type


def verify_flops(mod, expected_map):
    with Device("cpu"):
        ret = EstimateGFLOPS(run_infer_type(mod))
        ret = {k.name_hint: v.value for k, v in ret.items()}

    for var_name, expected_flops in expected_map.items():
        assert var_name in ret, "Missing %s" % var_name
        assert abs(expected_flops / 1e9 - ret[var_name]) <= 1e-2, "%s GFLOPS mismatch" % var_name


def test_conv2d():
    shape = (16, 16, 64, 64)

    def get_mod():
        conv2d_call = lambda x, w: raf.ir.op.conv2d(x, w, 1, 1)

        data = raf.ir.var("x", shape=shape)
        weight = raf.ir.var("w", shape=(16, 16, 3, 3))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", conv2d_call(data, weight))
        sb.ret(a_1)
        func = relay.Function([data, weight], sb.get())
        return tvm.IRModule.from_expr(func)

    # 2 * (N * CI * CO * H * W * kh * kw)
    verify_flops(get_mod(), {"a1": 2 * 16 * 16 * 16 * 64 * 64 * 3 * 3})


def test_unary():
    shape = (10, 5)

    def get_mod():
        data = raf.ir.var("x", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", raf.ir.op.relu(data))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1))
        sb.ret(a_2)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    verify_flops(get_mod(), {"a1": 10 * 5, "a2": 10 * 5})


def test_fusion():
    shape = (10, 5)

    def get_mod():
        data = raf.ir.var("x", shape=shape)

        p_0 = raf.ir.var("p0", shape=shape)
        out = raf.ir.op.relu(p_0)
        out = raf.ir.op.relu(out)
        closure = relay.Function([p_0], out)
        closure = closure.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", relay.Call(closure, [data]))
        sb.ret(a_1)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    verify_flops(get_mod(), {"a1": 10 * 5 * 2})


def test_multi_func():
    shape = (10, 5)

    def get_mod():
        sb = ScopeBuilder()
        data = raf.ir.var("x", shape=shape)
        a_1 = sb.let("a1", raf.ir.op.relu(data))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1))
        sb.ret(a_2)

        mod = tvm.IRModule()
        func_1 = relay.GlobalVar("func_1")
        mod[func_1] = relay.Function([data], sb.get())

        sb = ScopeBuilder()
        data = raf.ir.var("x", shape=shape)
        b_1 = sb.let("b1", relay.Call(func_1, [data]))
        sb.ret(b_1)
        mod[relay.GlobalVar("main")] = relay.Function([data], sb.get())
        return mod

    verify_flops(get_mod(), {"b1": 10 * 5 * 2})


if __name__ == "__main__":
    pytest.main([__file__])
