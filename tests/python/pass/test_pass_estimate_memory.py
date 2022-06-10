# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=protected-access
import pytest
import raf
import tvm
from tvm import relay

from raf._core.device import Device
from raf._core.vm import VMCompiler
from raf._ffi.pass_ import EstimateMemory, InferType
from raf.ir import ScopeBuilder
from raf.testing import check


def verify_memory(mod, device, expected_trace, disable_fusion=True, include_param=False):
    disabled_pass = []
    if disable_fusion:
        disabled_pass += ["FuseDialect", "FuseTVM"]

    compiler = VMCompiler()
    with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_pass):
        mod, _ = compiler.optimize(mod, device)
    mod = InferType()(mod)
    trace = [(name, mem.value) for name, mem in EstimateMemory(mod, Device(device), include_param)]
    assert len(trace) == len(expected_trace)
    for (name, mem), expected in zip(trace, expected_trace):
        assert name != "unknown"
        if isinstance(expected, tuple):  # The expected memory could be a range.
            assert expected[0] <= mem <= expected[1], f"{expected[0]} <= {mem} <= {expected[1]}"
        else:
            check(mem, expected)


def test_basic():
    shape = (512, 512)  # 1 MB
    device = "cpu"

    def get_mod():
        data = raf.ir.var("x", shape=shape)

        sb = ScopeBuilder()
        a_1 = sb.let("a1", raf.ir.op.relu(data))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1))
        a_3 = sb.let("a3", raf.ir.op.relu(a_2))
        sb.ret(a_3)
        func = relay.Function([data], sb.get())
        return tvm.IRModule.from_expr(func)

    verify_memory(get_mod(), device, [1, 2, 2, 1], True)  # Individual ops.
    verify_memory(get_mod(), device, [2, 3, 3, 2], True, True)  # Individual ops with parameters.
    verify_memory(get_mod(), device, [1, 1], False)  # Fused to one op.


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
def test_workspace():
    shape = (16, 16, 32, 32)  # 1 MB

    def get_mod():
        conv2d_call = lambda x, w: raf.ir.op.conv2d(x, w, 1, 1)

        data = raf.ir.var("x", shape=shape)
        weight = raf.ir.var("w", shape=(16, 16, 3, 3))

        sb = ScopeBuilder()
        a_1 = sb.let("a1", conv2d_call(data, weight))
        a_2 = sb.let("a2", raf.ir.op.relu(a_1))
        sb.ret(a_2)
        func = relay.Function([data, weight], sb.get())
        return tvm.IRModule.from_expr(func)

    # The memory at Conv2D should be 1 MB+workspace, but the workspace should be
    # freed afterward, so the following ReLU should only have 2 MBs.
    verify_memory(get_mod(), "cuda", [(1, float("inf")), 2, 1], True)


if __name__ == "__main__":
    pytest.main([__file__])
