# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,invalid-name,unused-variable,no-self-use,too-many-locals
import pytest
import numpy as np

import tvm
import raf
from raf.ir import RAFSequential, ScopeBuilder
from raf._ffi.pass_ import InferType, PartitionANF
from raf._core.executor import VMExecutor
from raf._core.ir_ext import extended_var
from raf._core.module import IRModule
from raf.testing import get_testable_devices, check


@pytest.mark.parametrize("device", get_testable_devices())
def test_diamond(device):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            r = raf.relu(x)
            a_1 = raf.abs(r)
            a_2 = raf.tanh(r)
            out = raf.add(a_1, a_2)
            return out

    shape = (10, 10)
    model = Model()
    x = raf.array(np.random.randn(*shape), dtype="float32")
    ref_z = model(x)

    mod = model._internal(x).mod
    seq = RAFSequential([InferType(), PartitionANF(2)])

    def expected():
        """
        fn(%x) {
          let %func_partition_0 = fn () {
            let %a1 = raf.op.relu(%x);
            let %a2 = raf.op.abs(%a1);
            let %func_partition_0_outs = (%a1, %a2);
            %func_partition_0_outs
          };
          let %func_partition_0_ret = %func_partition_0();
          let %func_partition_0_ret_0 = %func_partition_0_ret.0;
          let %func_partition_0_ret_1 = %func_partition_0_ret.1;
          let %func_partition_1 = fn () {
            let %a3 = raf.op.tanh(%func_partition_0_ret_0);
            let %a4 = raf.op.add(%func_partition_0_ret_1, %a3, None, None);
            %a4
          };
          let %func_partition_1_ret = %func_partition_1();
          %func_partition_1_ret
        }
        """
        builder = ScopeBuilder()
        x = extended_var("x", shape=shape)

        builder_func_0 = ScopeBuilder()
        a1 = builder_func_0.let("a1", raf.ir.op.relu(x))
        a2 = builder_func_0.let("a2", raf.ir.op.abs(a1))
        func_partition_0_outs = builder_func_0.let(
            "func_partition_0_outs", tvm.relay.Tuple([a1, a2])
        )
        builder_func_0.ret(func_partition_0_outs)
        func_0 = tvm.relay.Function([], builder_func_0.get())

        func_partition_0 = builder.let("func_partition_0", func_0)
        func_partition_0_ret = builder.let(
            "func_partition_0_ret", tvm.relay.Call(func_partition_0, [])
        )

        func_partition_0_ret_0 = builder.let("", tvm.relay.TupleGetItem(func_partition_0_ret, 0))
        func_partition_0_ret_1 = builder.let("", tvm.relay.TupleGetItem(func_partition_0_ret, 1))

        builder_func_1 = ScopeBuilder()
        a3 = builder_func_1.let("a3", raf.ir.op.tanh(func_partition_0_ret_0))
        a4 = builder_func_1.let("a4", raf.ir.op.add(func_partition_0_ret_1, a3))
        builder_func_1.ret(a4)
        func_1 = tvm.relay.Function([], builder_func_1.get())
        func_partition_1 = builder.let("func_partition_1", func_1)
        func_partition_1_ret = builder.let(
            "func_partition_1_ret", tvm.relay.Call(func_partition_1, [])
        )
        builder.ret(func_partition_1_ret)
        main = tvm.relay.Function([x], builder.get())
        return IRModule.from_expr(main)

    mod = seq(mod)
    expected_func = InferType()(expected())["main"]
    main_func = InferType()(mod)["main"]
    assert tvm.ir.structural_equal(main_func, expected_func)

    executor = VMExecutor(mod, device)
    m_z = executor.make_executor()(x)
    check(m_z, ref_z)


@pytest.mark.parametrize("device", get_testable_devices())
def test_tuple(device):
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            r = raf.relu(x)
            a_1 = raf.abs(r)
            a_2 = raf.tanh(r)
            out = raf.concatenate((a_1, a_2))
            out = raf.abs(out)
            return out

    shape = (10, 10)
    model = Model()
    x = raf.array(np.random.randn(*shape), dtype="float32")
    ref_z = model(x)

    mod = model._internal(x).mod

    def expected():
        """
        main(%x: Tensor[(10, 10), float32]) -> Tensor[(20, 10), float32] {
          let %func_partition_0 = fn () {
            let %a1 = raf.op.relu(%x);
            let %a2 = raf.op.abs(%a1);
            let %a3 = raf.op.tanh(%a1);
            let %a4 = (%a2, %a3);
             # TODO(yzhliu): we can improve further to remove %a2 and %a3
            let %func_partition_0_outs = (%a2, %a3, %a4);
            %func_partition_0_outs
          };
          let %func_partition_0_ret = %func_partition_0();
          let %func_partition_0_ret_0 = %func_partition_0_ret.0;
          let %func_partition_0_ret_1 = %func_partition_0_ret.1;
          let %func_partition_0_ret_2 = %func_partition_0_ret.2;
          let %func_partition_1 = fn () {
            let %a5 = raf.op.concatenate(%func_partition_0_ret_2, int64(0));
            let %a6 = raf.op.abs(%a5);
            %a6
          };
          let %func_partition_1_ret = %func_partition_1();
          %func_partition_1_ret
        }
        """
        builder = ScopeBuilder()
        x = extended_var("x", shape=shape)

        builder_func_0 = ScopeBuilder()
        a1 = builder_func_0.let("a1", raf.ir.op.relu(x))
        a2 = builder_func_0.let("a2", raf.ir.op.abs(a1))
        a3 = builder_func_0.let("a3", raf.ir.op.tanh(a1))
        a4 = builder_func_0.let("a4", tvm.relay.Tuple([a2, a3]))
        func_partition_0_outs = builder_func_0.let(
            "func_partition_0_outs", tvm.relay.Tuple([a2, a3, a4])
        )
        builder_func_0.ret(func_partition_0_outs)
        func_0 = tvm.relay.Function([], builder_func_0.get())

        func_partition_0 = builder.let("func_partition_0", func_0)
        func_partition_0_ret = builder.let(
            "func_partition_0_ret", tvm.relay.Call(func_partition_0, [])
        )

        func_partition_0_ret_0 = builder.let("", tvm.relay.TupleGetItem(func_partition_0_ret, 0))
        func_partition_0_ret_1 = builder.let("", tvm.relay.TupleGetItem(func_partition_0_ret, 1))
        func_partition_0_ret_2 = builder.let("", tvm.relay.TupleGetItem(func_partition_0_ret, 2))

        builder_func_1 = ScopeBuilder()
        a5 = builder_func_1.let(
            "a5", raf.ir.op.concatenate(func_partition_0_ret_2, raf.ir.const(0))
        )
        a6 = builder_func_1.let("a6", raf.ir.op.abs(a5))
        builder_func_1.ret(a6)
        func_1 = tvm.relay.Function([], builder_func_1.get())
        func_partition_1 = builder.let("func_partition_1", func_1)
        func_partition_1_ret = builder.let(
            "func_partition_1_ret", tvm.relay.Call(func_partition_1, [])
        )
        builder.ret(func_partition_1_ret)
        main = tvm.relay.Function([x], builder.get())
        return IRModule.from_expr(main)

    seq = RAFSequential([InferType(), PartitionANF(3)])
    mod = seq(mod)

    expected_func = InferType()(expected())["main"]
    main_func = InferType()(mod)["main"]
    assert tvm.ir.structural_equal(main_func, expected_func)

    executor = VMExecutor(mod, device)
    m_z = executor.make_executor()(x)
    check(m_z, ref_z)


if __name__ == "__main__":
    pytest.main([__file__])
