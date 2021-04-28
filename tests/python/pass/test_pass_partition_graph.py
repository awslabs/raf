# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm.ir import MNMSequential
from mnm._ffi.pass_ import AnnotateTarget, MergeCompilerRegions
from mnm._ffi.pass_ import PartitionGraph, InferType
from mnm._core.ir_ext import extended_var
from mnm._lib import tvm
from mnm._lib import relay as _relay
from mnm._core.module import IRModule


def test_diamond():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    # pylint: disable=too-many-statements
    target = "test_diamond"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.abs", "target." + target)
    def abs(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.add", "target." + target)
    def add(attrs, args): # pylint: disable=unused-argument
        return True

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            r = mnm.relu(x)
            a_1 = mnm.abs(r)
            a_2 = mnm.tanh(r)
            out = mnm.add(a_1, a_2)
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = fn (global_symbol="test_diamond_0", Compiler="test_diamond") {
        #     let %test_diamond_0_0 = mnm.op.relu(%x);
        #     let %test_diamond_0_1 = mnm.op.abs(%test_diamond_0_0);
        #     let %test_diamond_0_outs = (%test_diamond_0_0, %test_diamond_0_1);
        #     %test_diamond_0_outs
        # };
        # let %test_diamond_0_ret = %0();
        # let %a1 = %test_diamond_0_ret.0;
        # let %a2 = %test_diamond_0_ret.1;
        # let %a3 = mnm.op.tanh(%a1);
        # let %a4 = mnm.op.add(%a2, %a3);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        test_diamond_0_0 = extended_var("test_diamond_0_0")
        test_diamond_0_1 = extended_var("test_diamond_0_1")
        test_diamond_0_outs = extended_var("test_diamond_0_outs")
        test_diamond_0_ret = extended_var("test_diamond_0_ret")
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        tanh = _relay.op.get("mnm.op.tanh")
        add = _relay.op.get("mnm.op.add")
        # define functions
        relu_call = _relay.Call(relu, [x])
        abs_call = _relay.Call(abs, [test_diamond_0_0])
        func1_out_tuple = _relay.Tuple([test_diamond_0_0, test_diamond_0_1])
        func1_body = _relay.Let(test_diamond_0_outs, func1_out_tuple, test_diamond_0_outs)
        func1_body = _relay.Let(test_diamond_0_1, abs_call, func1_body)
        func1_body = _relay.Let(test_diamond_0_0, relu_call, func1_body)
        func1 = _relay.Function([], func1_body)
        func1 = func1.with_attr("global_symbol", "test_diamond_0")
        func1 = func1.with_attr("Compiler", "test_diamond")
        # define function calls and tuple get items
        func1_call = _relay.Call(func1, [])
        func1_tgi_0 = _relay.TupleGetItem(test_diamond_0_ret, 0)
        func1_tgi_1 = _relay.TupleGetItem(test_diamond_0_ret, 1)
        # make anf
        tanh_call = _relay.Call(tanh, [a1])
        add_call = _relay.Call(add, [a2, a3])
        body = _relay.Let(a4, add_call, a4)
        body = _relay.Let(a3, tanh_call, body)
        body = _relay.Let(a2, func1_tgi_1, body)
        body = _relay.Let(a1, func1_tgi_0, body)
        body = _relay.Let(test_diamond_0_ret, func1_call, body)
        func = _relay.Function([x], body)
        return IRModule.from_expr(func)

    # annotate ir and merge compiler regions
    model = Model()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    seq = MNMSequential([AnnotateTarget([target]),
                         MergeCompilerRegions(), PartitionGraph(), InferType()])
    mod = seq(mod)
    func = mod["main"]
    expected_mod = expected()
    expected_mod = InferType()(expected_mod)
    expected_func = expected_mod["main"]

    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


def test_tuple():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable, too-many-statements
    target1 = "test_tuple1"
    target2 = "test_tuple2"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target1)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.abs", "target." + target1)
    def abs(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.tanh", "target." + target2)
    def tanh(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.concatenate", "target." + target2)
    def concatenate(attrs, args): # pylint: disable=unused-argument
        return True

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            r = mnm.relu(x)
            a_1 = mnm.abs(r)
            a_2 = mnm.tanh(r)
            out = mnm.concatenate((a_1, a_2))
            out = mnm.abs(out)
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = fn (global_symbol="test_tuple1_0", Compiler="test_tuple1") {
        #     let %test_tuple1_0_0 = mnm.op.relu(%x);
        #     let %test_tuple1_0_1 = mnm.op.abs(%test_tuple1_0_0);
        #     let %test_tuple1_0_outs = (%test_tuple1_0_0, %test_tuple1_0_1);
        #     %test_tuple1_0_outs
        # };
        # let %test_tuple1_0_ret = %0();
        # let %a1 = %test_tuple1_0_ret.0;
        # let %a2 = %test_tuple1_0_ret.1;
        # %1 = fn (global_symbol="test_tuple2_0", Compiler="test_tuple2") {
        #     let %test_tuple2_0_0 = mnm.op.tanh(%a1);
        #     let %test_tuple2_0_1 = (%a2, %test_tuple2_0_0);
        #     let %test_tuple2_0_2 = mnm.op.concatenate(%test_tuple2_0_1, -114514);
        #     let %test_tuple2_0_outs = (%test_tuple2_0_0, %test_tuple2_0_1, %test_tuple2_0_2);
        #     %test_tuple2_0_outs
        # };
        # let %test_tuple2_0_ret = %1();
        # let %a3 = %test_tuple2_0_ret.0;
        # let %a4 = %test_tuple2_0_ret.1;
        # let %a5 = %test_tuple2_0_ret.2;
        # let %a6 = mnm.op.abs(%a5);
        # %a6
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        a5 = extended_var("a5")
        a6 = extended_var("a6")
        test_tuple1_0_0 = extended_var("test_tuple1_0_0")
        test_tuple1_0_1 = extended_var("test_tuple1_0_1")
        test_tuple1_0_outs = extended_var("test_tuple1_0_outs")
        test_tuple1_0_ret = extended_var("test_tuple1_0_ret")
        test_tuple2_0_0 = extended_var("test_tuple2_0_0")
        test_tuple2_0_1 = extended_var("test_tuple2_0_1")
        test_tuple2_0_2 = extended_var("test_tuple2_0_2")
        test_tuple2_0_outs = extended_var("test_tuple2_0_outs")
        test_tuple2_0_ret = extended_var("test_tuple2_0_ret")
        const = mnm.ir.const(0)
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        tanh = _relay.op.get("mnm.op.tanh")
        concat = _relay.op.get("mnm.op.concatenate")
        # define functions
        # func1
        relu_call = _relay.Call(relu, [x])
        abs_call1 = _relay.Call(abs, [test_tuple1_0_0])
        func1_out_tuple = _relay.Tuple([test_tuple1_0_0, test_tuple1_0_1])
        func1_body = _relay.Let(test_tuple1_0_outs, func1_out_tuple, test_tuple1_0_outs)
        func1_body = _relay.Let(test_tuple1_0_1, abs_call1, func1_body)
        func1_body = _relay.Let(test_tuple1_0_0, relu_call, func1_body)
        func1 = _relay.Function([], func1_body)
        func1 = func1.with_attr("global_symbol", "test_tuple1_0")
        func1 = func1.with_attr("Compiler", "test_tuple1")
        # func2
        tanh_call = _relay.Call(tanh, [a1])
        concat_tuple = _relay.Tuple([a2, test_tuple2_0_0])
        concat_call = _relay.Call(concat, [test_tuple2_0_1, const])
        func2_out_tuple = _relay.Tuple([test_tuple2_0_0, test_tuple2_0_1, test_tuple2_0_2])
        func2_body = _relay.Let(test_tuple2_0_outs, func2_out_tuple, test_tuple2_0_outs)
        func2_body = _relay.Let(test_tuple2_0_2, concat_call, func2_body)
        func2_body = _relay.Let(test_tuple2_0_1, concat_tuple, func2_body)
        func2_body = _relay.Let(test_tuple2_0_0, tanh_call, func2_body)
        func2 = _relay.Function([], func2_body)
        func2 = func2.with_attr("global_symbol", "test_tuple2_0")
        func2 = func2.with_attr("Compiler", "test_tuple2")
        # define function calls and tuple get items
        # func1 call
        func1_call = _relay.Call(func1, [])
        func1_tgi_0 = _relay.TupleGetItem(test_tuple1_0_ret, 0)
        func1_tgi_1 = _relay.TupleGetItem(test_tuple1_0_ret, 1)
        # func2 call
        func2_call = _relay.Call(func2, [])
        func2_tgi_0 = _relay.TupleGetItem(test_tuple2_0_ret, 0)
        func2_tgi_1 = _relay.TupleGetItem(test_tuple2_0_ret, 1)
        func2_tgi_2 = _relay.TupleGetItem(test_tuple2_0_ret, 2)
        # make anf
        abs_call2 = _relay.Call(abs, [a5])
        body = _relay.Let(a6, abs_call2, a6)
        body = _relay.Let(a5, func2_tgi_2, body)
        body = _relay.Let(a4, func2_tgi_1, body)
        body = _relay.Let(a3, func2_tgi_0, body)
        body = _relay.Let(test_tuple2_0_ret, func2_call, body)
        body = _relay.Let(a2, func1_tgi_1, body)
        body = _relay.Let(a1, func1_tgi_0, body)
        body = _relay.Let(test_tuple1_0_ret, func1_call, body)
        func = _relay.Function([x], body)
        return IRModule.from_expr(func)

    # annotate ir and merge compiler regions
    model = Model()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    seq = MNMSequential([AnnotateTarget([target1, target2]),
                         MergeCompilerRegions(), PartitionGraph(), InferType()])
    mod = seq(mod)
    func = mod["main"]
    expected_mod = expected()
    expected_mod = InferType()(expected_mod)
    expected_func = expected_mod["main"]
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
