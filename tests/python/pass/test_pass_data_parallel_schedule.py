# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use
from typing import Dict, List
import pytest
import tvm

import raf
from raf.ir.pass_manager import RAFSequential
from raf._ffi.pass_ import DataParallelSchedule, ToGraphNormalForm
from raf.testing import randn
from raf._core.ir_ext import extended_var
from raf.ir import ScopeBuilder


class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators: Dict[str, tvm.ir.Op] = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = raf._ffi.op.GetOp(f"raf.op.{op_name}")
        return self.operators[op_name]

    def const(self, value):
        return raf.ir.const(value)

    def make_tuple(self, fields):
        return self.scope_builder.let("", tvm.relay.Tuple(fields))

    def get_tuple_item(self, tup, index):
        return self.scope_builder.let("", tvm.relay.TupleGetItem(tup, index))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let("", tvm.relay.Call(self.get_operator(op_name), args))

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()


class TwoBranchModel(raf.Model):
    #        /-> atan -> allreduce -> mul -\
    #  -> mul                               concat ->
    #        \-> atan -> allreduce -> mul -/
    def build(self, shape):
        self.shape = shape
        self.c, _ = randn(shape, device="cuda", requires_grad=True)

    @raf.model.trace
    def forward(self, x):
        a0 = raf.multiply(x, self.c)
        a1_a = raf.atan(a0)
        a1_b = raf.atan(a0)
        a2_a = raf.allreduce(a1_a)
        a2_b = raf.allreduce(a1_b)
        a3_a = raf.multiply(a2_a, self.c)
        a3_b = raf.multiply(a2_b, self.c)
        a4 = raf.concatenate([a3_a, a3_b])
        return a4

    def fifo_expected(self):
        """
        fn (%x: Tensor[(64, 128), float32], %c: Tensor[(64, 128), float32]) {
            let %v = raf.op.multiply(%x, %c);
            let %v1 = raf.op.atan(%v);
            let %v2 = raf.op.atan(%v);
            let %v3 = (%v1,);
            let %v4 = (%v2,);
            let %v5 = raf.op._allreduce(%v3, str"sum", nullptr);
            let %v6 = raf.op._allreduce(%v4, str"sum", nullptr);
            let %v7 = raf.op.multiply(%v5, %c);
            let %v8 = raf.op.multiply(%v6, %c);
            let %v9 = (%v7, %v8);
            let %v10 = raf.op.concatenate(%v9, int64(0));
            %v10
        }
        """
        builder = ANFBuilder()
        x = extended_var("x", shape=self.shape)
        c = extended_var("c", shape=self.shape)

        x_0 = builder.call("multiply", [x, c])
        x_1a = builder.call("atan", [x_0])
        x_1b = builder.call("atan", [x_0])

        x_2a = builder.make_tuple((x_1a,))
        x_2b = builder.make_tuple((x_1b,))

        x_2a1 = builder.const("sum")
        x_3a = builder.call("_allreduce", [x_2a, x_2a1, builder.const(None)])
        x_2a2 = builder.const("sum")
        x_3b = builder.call("_allreduce", [x_2b, x_2a2, builder.const(None)])

        x_4a = builder.call("multiply", [x_3a, c])
        x_4b = builder.call("multiply", [x_3b, c])

        x_5 = builder.make_tuple([x_4a, x_4b])
        x_5_1 = builder.const(0)

        x_6 = builder.call("concatenate", [x_5, x_5_1])
        return [tvm.relay.Function([x, c], builder.ret(x_6))]


class UnbalancedModel(raf.Model):
    #        /-> atan -> atan -> atan -> atan -> allreduce -> mul -\
    #  -> mul                                                       concat ->
    #        \-> atan -> allreduce -> mul -------------------------/
    def build(self, shape):
        self.shape = shape
        self.c, _ = randn(shape, device="cuda", requires_grad=True)

    @raf.model.trace
    def forward(self, x):
        a0 = raf.multiply(x, self.c)
        a1_a = raf.atan(a0)
        a2_a = raf.atan(a1_a)
        a3_a = raf.atan(a2_a)
        a4_a = raf.atan(a3_a)
        a5_a = raf.allreduce(a4_a)
        a6_a = raf.multiply(a5_a, self.c)

        a1_b = raf.atan(a0)
        a2_b = raf.allreduce(a1_b)
        a3_b = raf.multiply(a2_b, self.c)

        a4 = raf.concatenate([a6_a, a3_b])
        return a4

    def fifo_expected(self):
        """(version 1)
        fn (%x: Tensor[(64, 128), float32], %c: Tensor[(64, 128), float32]) {
            let %v = raf.op.multiply(%x, %c);
            let %v1 = raf.op.atan(%v);
            let %v2 = raf.op.atan(%v);
            let %v3 = raf.op.atan(%v1);
            let %v4 = (%v2,);
            let %v5 = raf.op.atan(%v3);
            let %v6 = raf.op._allreduce(%v4, str"sum", nullptr);
            let %v7 = raf.op.atan(%v5);
            let %v8 = (%v7,);
            let %v9 = raf.op._allreduce(%v8, str"sum", nullptr);
            let %v10 = raf.op.multiply(%v6, %c);
            let %v11 = raf.op.multiply(%v9, %c);
            let %v12 = (%v11, %v10);
            let %v13 = raf.op.concatenate(%v12, int64(0));
            %v13
        }
        """

        def version1():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)
            c = extended_var("c", shape=self.shape)

            x_0 = builder.call("multiply", [x, c])
            # x_1a (upper branch) gets selected first
            x_1a = builder.call("atan", [x_0])
            x_1b = builder.call("atan", [x_0])

            x_2a = builder.call("atan", [x_1a])
            x_2b = builder.make_tuple((x_1b,))

            x_3a = builder.call("atan", [x_2a])
            x_2b1 = builder.const("sum")
            x_3b = builder.call("_allreduce", [x_2b, x_2b1, builder.const(None)])

            x_4a = builder.call("atan", [x_3a])
            # branch b is delayed since mul depdends on allreduce
            x_5a = builder.make_tuple((x_4a,))
            x_5a1 = builder.const("sum")
            x_6a = builder.call("_allreduce", [x_5a, x_5a1, builder.const(None)])

            # now launch update ops, branch b first
            x_4b = builder.call("multiply", [x_3b, c])
            x_7a = builder.call("multiply", [x_6a, c])

            x_8 = builder.make_tuple((x_7a, x_4b))

            x_8a = builder.const(0)
            x_9 = builder.call("concatenate", [x_8, x_8a])
            return tvm.relay.Function([x, c], builder.ret(x_9))

        def version2():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)
            c = extended_var("c", shape=self.shape)

            x_0 = builder.call("multiply", [x, c])
            # x_1b (upper branch) gets selected first
            x_1b = builder.call("atan", [x_0])
            x_1a = builder.call("atan", [x_0])

            x_2b = builder.make_tuple((x_1b,))
            x_2a = builder.call("atan", [x_1a])

            x_2b1 = builder.const("sum")
            x_3b = builder.call("_allreduce", [x_2b, x_2b1, builder.const(None)])
            x_3a = builder.call("atan", [x_2a])

            # branch b is delayed since mul depdends on allreduce
            x_4a = builder.call("atan", [x_3a])
            x_5a = builder.make_tuple((x_4a,))
            x_5a1 = builder.const("sum")
            x_6a = builder.call("_allreduce", [x_5a, x_5a1, builder.const(None)])

            # now launch update ops, branch b first
            x_4b = builder.call("multiply", [x_3b, c])
            x_7a = builder.call("multiply", [x_6a, c])

            x8 = builder.make_tuple((x_7a, x_4b))

            x_8a = builder.const(0)
            x9 = builder.call("concatenate", [x8, x_8a])
            return tvm.relay.Function([x, c], builder.ret(x9))

        return [version1(), version2()]


class ExampleModel(raf.Model):
    # the example model where ToANF can generate bad schedule:
    #         /-> allreduce -> atan -\
    #  -> atan ->    atan   -> atan -> mul
    def build(self, shape):
        self.shape = shape

    @raf.model.trace
    def forward(self, x):
        a0 = raf.atan(x)
        a1_a = raf.allreduce(a0)
        a2_a = raf.atan(a1_a)

        a1_b = raf.atan(a0)
        a2_b = raf.atan(a1_b)

        a3 = raf.multiply(a2_a, a2_b)
        return a3

    def fifo_expected(self):
        """(version 1)
        fn (%x: Tensor[(64, 128), float32]) {
            let %v = raf.op.atan(%x);
            let %v1 = (%v,);
            let %v2 = raf.op.atan(%v);
            let %v3 = raf.op._allreduce(%v1, str"sum", nullptr);
            let %v4 = raf.op.atan(%v2);
            let %v5 = raf.op.atan(%v3);
            let %v6 = raf.op.multiply(%v5, %v4);
            %v6
        }
        """

        def version1():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)

            a0 = builder.call("atan", [x])

            a1_ai = builder.make_tuple((a0,))
            a1_b = builder.call("atan", [a0])

            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])
            a2_b = builder.call("atan", [a1_b])

            a2_a = builder.call("atan", [a1_a])

            a3 = builder.call("multiply", [a2_a, a2_b])
            return tvm.relay.Function([x], builder.ret(a3))

        def version2():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)

            a0 = builder.call("atan", [x])

            a1_b = builder.call("atan", [a0])
            a1_ai = builder.make_tuple((a0,))

            a2_b = builder.call("atan", [a1_b])
            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])

            a2_a = builder.call("atan", [a1_a])

            a3 = builder.call("multiply", [a2_a, a2_b])
            return tvm.relay.Function([x], builder.ret(a3))

        return [version1(), version2()]


class DelayedSuccessorModel(raf.Model):
    #         /-> allreduce -> relu -> relu -----------------\
    #  -> atan ->    atan   -> atan -> atan -> atan -> atan -> mul
    def build(self, shape):
        self.shape = shape

    @raf.model.trace
    def forward(self, x):
        a0 = raf.atan(x)
        a1_a = raf.allreduce(a0)
        a2_a = raf.relu(a1_a)
        a3_a = raf.relu(a2_a)

        a1_b = raf.atan(a0)
        a2_b = raf.atan(a1_b)
        a3_b = raf.atan(a2_b)
        a4_b = raf.atan(a3_b)
        a5_b = raf.atan(a4_b)

        a6 = raf.multiply(a3_a, a5_b)
        return a6

    def fifo_expected(self):
        """(version 1)
        fn (%x: Tensor[(64, 128), float32]) {
            let %x_0 = raf.op.atan(%x);
            let %x_1 = (%x_0,);
            let %x_2 = raf.op.atan(%x_0);
            let %x_3 = raf.op._allreduce(%x_1, str"sum", nullptr);
            let %x_4 = raf.op.atan(%x_2);
            let %x_5 = raf.op.atan(%x_4);
            let %x_6 = raf.op.atan(%x_5);
            let %x_7 = raf.op.atan(%x_6);
            let %x_8 = raf.op.relu(%x_3);
            let %x_9 = raf.op.relu(%x_8);
            let %x_10 = raf.op.multiply(%x_9, %x_7);
            %x_10
        }
        """

        def version1():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)

            a0 = builder.call("atan", [x])

            a1_ai = builder.make_tuple((a0,))
            a1_b = builder.call("atan", [a0])

            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])
            a2_b = builder.call("atan", [a1_b])

            a3_b = builder.call("atan", [a2_b])
            a4_b = builder.call("atan", [a3_b])
            a5_b = builder.call("atan", [a4_b])

            a2_a = builder.call("relu", [a1_a])
            a3_a = builder.call("relu", [a2_a])

            a6 = builder.call("multiply", [a3_a, a5_b])
            return tvm.relay.Function([x], builder.ret(a6))

        def version2():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)

            a0 = builder.call("atan", [x])

            a1_b = builder.call("atan", [a0])
            a1_ai = builder.make_tuple((a0,))

            a2_b = builder.call("atan", [a1_b])
            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])

            a3_b = builder.call("atan", [a2_b])
            a4_b = builder.call("atan", [a3_b])
            a5_b = builder.call("atan", [a4_b])

            a2_a = builder.call("relu", [a1_a])
            a3_a = builder.call("relu", [a2_a])

            a6 = builder.call("multiply", [a3_a, a5_b])
            return tvm.relay.Function([x], builder.ret(a6))

        return [version1(), version2()]


class NestedBranchModel(raf.Model):
    #                           /-> atan -\
    #        /-> atan -> atan ->           mul--> mul ->
    #  -> mul                   \-> atan -/    /
    #        \-> atan -> atan ----> atan ---->/
    def build(self, shape):
        self.shape = shape
        self.c, _ = randn(shape, device="cuda", requires_grad=True)

    @raf.model.trace
    def forward(self, x):
        a0 = raf.multiply(x, self.c)

        a1_a = raf.atan(a0)
        a2_a = raf.atan(a1_a)

        a3_a = raf.atan(a2_a)
        a4_a = raf.atan(a2_a)
        a5_a = raf.multiply(a3_a, a4_a)

        a1_b = raf.atan(a0)
        a2_b = raf.atan(a1_b)
        a3_b = raf.atan(a2_b)

        a6 = raf.multiply(a5_a, a3_b)
        return a6

    def fifo_expected(self):
        """(version 1)
        fn (%x: Tensor[(64, 128), float32], %c: Tensor[(64, 128), float32]) {
            let %v = raf.op.multiply(%x, %c);
            let %v1 = raf.op.atan(%v);
            let %v2 = raf.op.atan(%v);
            let %v3 = raf.op.atan(%v1);
            let %v4 = raf.op.atan(%v2);
            let %v5 = raf.op.atan(%v3);
            let %v6 = raf.op.atan(%v3);
            let %v7 = raf.op.atan(%v4);
            let %v8 = raf.op.multiply(%v5, %v6);
            let %v9 = raf.op.multiply(%v8, %v7);
            %v9
        }
        """

        def version1():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)
            c = extended_var("c", shape=self.shape)

            x_0 = builder.call("multiply", [x, c])
            # x_1a (upper branch) gets selected first
            x_1a = builder.call("atan", [x_0])
            x_1b = builder.call("atan", [x_0])

            x_2a = builder.call("atan", [x_1a])
            x_2b = builder.call("atan", [x_1b])

            x_3a1 = builder.call("atan", [x_2a])
            x_3a2 = builder.call("atan", [x_2a])
            x_3b = builder.call("atan", [x_2b])

            x_4a = builder.call("multiply", [x_3a1, x_3a2])
            x_5 = builder.call("multiply", [x_4a, x_3b])
            return tvm.relay.Function([x, c], builder.ret(x_5))

        def version2():
            builder = ANFBuilder()
            x = extended_var("x", shape=self.shape)
            c = extended_var("c", shape=self.shape)

            x_0 = builder.call("multiply", [x, c])
            # x_1a (upper branch) gets selected first
            x_1b = builder.call("atan", [x_0])
            x_1a = builder.call("atan", [x_0])

            x_2b = builder.call("atan", [x_1b])
            x_2a = builder.call("atan", [x_1a])

            x_3b = builder.call("atan", [x_2b])
            x_3a1 = builder.call("atan", [x_2a])
            x_3a2 = builder.call("atan", [x_2a])

            x_4a = builder.call("multiply", [x_3a1, x_3a2])
            x_5 = builder.call("multiply", [x_4a, x_3b])
            return tvm.relay.Function([x, c], builder.ret(x_5))

        return [version1(), version2()]


class CascadingCollectiveModel(raf.Model):
    #        /-> allreduce -> relu ---------\     /-> allreduce -> relu ---------\
    #  -> mul ->    atan   -> atan -> atan -> mul  ->    atan   -> atan -> atan -> mul
    def build(self, shape):
        self.shape = shape
        self.c, _ = randn(shape, device="cuda", requires_grad=True)

    @raf.model.trace
    def forward(self, x):
        a0 = raf.multiply(x, self.c)

        a1_a = raf.allreduce(a0)
        a2_a = raf.relu(a1_a)

        a1_b = raf.atan(a0)
        a2_b = raf.atan(a1_b)
        a3_b = raf.atan(a2_b)

        a4 = raf.multiply(a2_a, a3_b)

        a5_a = raf.allreduce(a4)
        a6_a = raf.relu(a5_a)

        a5_b = raf.atan(a4)
        a6_b = raf.atan(a5_b)
        a7_b = raf.atan(a6_b)

        a8 = raf.multiply(a6_a, a7_b)
        return a8

    def fifo_expected(self):
        """(version 1, version 1)
        fn (%x: Tensor[(64, 128), float32], %c: Tensor[(64, 128), float32]) {
            let %v = raf.op.multiply(%x, %c);
            let %v1 = (%v,);
            let %v2 = raf.op.atan(%v);
            let %v3 = raf.op._allreduce(%v1, str"sum", nullptr);
            let %v4 = raf.op.atan(%v2);
            let %v5 = raf.op.atan(%v4);
            let %v6 = raf.op.relu(%v3);
            let %v7 = raf.op.multiply(%v6, %v5);
            let %v8 = (%v7,);
            let %v9 = raf.op.atan(%v7);
            let %v10 = raf.op._allreduce(%v8, str"sum", nullptr);
            let %v11 = raf.op.atan(%v9);
            let %v12 = raf.op.atan(%v11);
            let %v13 = raf.op.relu(%v10);
            let %v14 = raf.op.multiply(%v13, %v12);
            %v14
        }
        """

        def version1(builder, input0, input1):
            a0 = builder.call("multiply", [input0, input1])

            a1_ai = builder.make_tuple((a0,))
            a1_b = builder.call("atan", [a0])

            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])
            a2_b = builder.call("atan", [a1_b])

            # a2_a is delayed after a3_b
            a3_b = builder.call("atan", [a2_b])
            a2_a = builder.call("relu", [a1_a])

            return a2_a, a3_b

        def version2(builder, input0, input1):
            a0 = builder.call("multiply", [input0, input1])

            a1_b = builder.call("atan", [a0])
            a1_ai = builder.make_tuple((a0,))

            a2_b = builder.call("atan", [a1_b])
            a1_aii = builder.const("sum")
            a1_a = builder.call("_allreduce", [a1_ai, a1_aii, builder.const(None)])

            # a2_a is delayed after a3_b
            a3_b = builder.call("atan", [a2_b])
            a2_a = builder.call("relu", [a1_a])

            return a2_a, a3_b

        candidate_irs = []
        for first_block in [version1, version2]:
            for second_block in [version1, version2]:
                anf_builder = ANFBuilder()
                x = extended_var("x", shape=self.shape)
                c = extended_var("c", shape=self.shape)
                b1_0, b1_1 = first_block(anf_builder, x, c)
                b2_0, b2_1 = second_block(anf_builder, b1_0, b1_1)
                res = anf_builder.call("multiply", [b2_0, b2_1])
                candidate_irs.append(tvm.relay.Function([x, c], anf_builder.ret(res)))

        return candidate_irs


@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "model_class,shape",
    [
        (TwoBranchModel, (64, 128)),
        (UnbalancedModel, (64, 128)),
        (ExampleModel, (64, 128)),
        (NestedBranchModel, (64, 128)),
        (DelayedSuccessorModel, (64, 128)),
        (CascadingCollectiveModel, (64, 128)),
    ],
)
def test_fifo_schedule(model_class, shape):
    model = model_class(shape)
    x, _ = randn(shape)
    mod = model._internal(x).mod

    mod = RAFSequential([ToGraphNormalForm(), DataParallelSchedule()])(mod)

    err_msgs = []
    err_msgs.append("Actual" + "<<" * 20)
    err_msgs.append(raf.ir.AsText(mod["main"]))

    equal_to_any = False
    for expected in model.fifo_expected():
        result = tvm.ir.structural_equal(mod["main"], expected)
        equal_to_any = equal_to_any or result
        err_msgs.append("Expected Candidate" + "<<" * 20)
        err_msgs.append(raf.ir.AsText(expected))
    assert equal_to_any, "\n".join(err_msgs)


if __name__ == "__main__":
    pytest.main([__file__])
