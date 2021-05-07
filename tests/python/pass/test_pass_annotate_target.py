# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm._ffi.pass_ import AnnotateTarget
from mnm._core.ir_ext import extended_var
from mnm._lib import tvm
from mnm._lib import relay as _relay


def test_multiple_ends():
    # pylint: disable=invalid-name, no-self-use, redefined-builtin, too-many-locals, unused-variable
    @tvm.ir.register_op_attr("mnm.op.relu", "target.test")
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            r = mnm.relu(x)
            a_1 = mnm.abs(r)
            a_2 = mnm.abs(r)
            out = mnm.add(a_1, a_2)
            return out

    def expected():
        # build the expected ir after annotated
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = mnm.op.compiler_begin(%x, meta[mnm.args.compiler][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[mnm.args.compiler][1]);
        # %2 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][2]);
        # %3 = mnm.op.abs(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[mnm.args.compiler][3]);
        # %4 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][4]);
        # %5 = mnm.op.abs(%4);
        # let %a3 = mnm.op.compiler_end(%5, meta[mnm.args.compiler][5]);
        # %6 = mnm.op.compiler_begin(%a2, meta[mnm.args.compiler][6]);
        # %7 = mnm.op.compiler_begin(%a3, meta[mnm.args.compiler][7]);
        # %8 = mnm.op.compiler_begin(nullptr, meta[mnm.args.compiler][8]);
        # %9 = mnm.op.compiler_begin(nullptr, meta[mnm.args.compiler][9]);
        # %10 = mnm.op.add(%6, %7, %8, %9);
        # let %a4 = mnm.op.compiler_end(%10, meta[mnm.args.compiler][10]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        null = mnm.ir.const(None)
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        add = _relay.op.get("mnm.op.add")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call = _relay.Call(begin, [x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call = _relay.Call(relu, [relu_call])
        relu_call = _relay.Call(end, [relu_call], tvm.ir.make_node("mnm.args.compiler"))
        abs_call1 = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        abs_call1 = _relay.Call(abs, [abs_call1])
        abs_call1 = _relay.Call(end, [abs_call1], tvm.ir.make_node("mnm.args.compiler"))
        abs_call2 = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        abs_call2 = _relay.Call(abs, [abs_call2])
        abs_call2 = _relay.Call(end, [abs_call2], tvm.ir.make_node("mnm.args.compiler"))
        let_call2 = _relay.Call(begin, [a2], tvm.ir.make_node("mnm.args.compiler"))
        let_call3 = _relay.Call(begin, [a3], tvm.ir.make_node("mnm.args.compiler"))
        const_call1 = _relay.Call(begin, [null], tvm.ir.make_node("mnm.args.compiler"))
        const_call2 = _relay.Call(begin, [null], tvm.ir.make_node("mnm.args.compiler"))
        add_call = _relay.Call(add, [let_call2, let_call3, const_call1, const_call2])
        add_call = _relay.Call(end, [add_call], tvm.ir.make_node("mnm.args.compiler"))
        # make anf
        body = _relay.Let(a4, add_call, a4)
        body = _relay.Let(a3, abs_call2, body)
        body = _relay.Let(a2, abs_call1, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate the ir with annotate_target pass
    model = Model()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget(["test"])(mod)
    expected_func = expected()
    # check the structure of the expected ir and generated ir
    assert tvm.ir.structural_equal(mod["main"], expected_func)


def test_tuple():
    # pylint: disable=invalid-name, no-self-use, too-many-locals, unused-variable
    target = "test_tuple_annotation"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.concatenate", "target." + target)
    def concatenate(attrs, args): # pylint: disable=unused-argument
        return True

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y):
            a_1 = mnm.relu(x)
            a_2 = mnm.relu(y)
            out = mnm.concatenate((a_1, a_2), axis=1)
            return out

    def expected():
        # build the expected ir after annotated
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64], %y: Tensor[(10, 10), float64]) {
        # %0 = mnm.op.compiler_begin(%x, meta[mnm.args.compiler][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[mnm.args.compiler][1]);
        # %2 = mnm.op.compiler_begin(%y, meta[mnm.args.compiler][2]);
        # %3 = mnm.op.relu(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[mnm.args.compiler][3]);
        # let %a3 = (%a1, %a2);
        # %4 = mnm.op.compiler_begin(%a3, meta[mnm.args.compiler][4]);
        # %5 = mnm.op.compiler_begin(-114514, meta[mnm.args.compiler][5]);
        # %6 = mnm.op.concatenate(%4, %5);
        # let %a4 = mnm.op.compiler_end(%6, meta[mnm.args.compiler][6]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        y = extended_var("y", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        const = _relay.Constant(tvm.nd.array(-114514))
        relu = _relay.op.get("mnm.op.relu")
        concatenate = _relay.op.get("mnm.op.concatenate")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call_x = _relay.Call(begin, [x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call_x = _relay.Call(relu, [relu_call_x])
        relu_call_x = _relay.Call(end, [relu_call_x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call_y = _relay.Call(begin, [y], tvm.ir.make_node("mnm.args.compiler"))
        relu_call_y = _relay.Call(relu, [relu_call_y])
        relu_call_y = _relay.Call(end, [relu_call_y], tvm.ir.make_node("mnm.args.compiler"))
        relu_tuple1 = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        relu_tuple2 = _relay.Call(begin, [a2], tvm.ir.make_node("mnm.args.compiler"))
        relu_tuple = _relay.Tuple([relu_tuple1, relu_tuple2])
        relu_tuple = _relay.Call(end, [relu_tuple], tvm.ir.make_node("mnm.args.compiler"))
        concat_call1 = _relay.Call(begin, [a3], tvm.ir.make_node("mnm.args.compiler"))
        concat_call2 = _relay.Call(begin, [const], tvm.ir.make_node("mnm.args.compiler"))
        concat_call = _relay.Call(concatenate, [concat_call1, concat_call2])
        concat_call = _relay.Call(end, [concat_call], tvm.ir.make_node("mnm.args.compiler"))
        # make anf
        body = _relay.Let(a4, concat_call, a4)
        body = _relay.Let(a3, relu_tuple, body)
        body = _relay.Let(a2, relu_call_y, body)
        body = _relay.Let(a1, relu_call_x, body)
        func = _relay.Function([x, y], body)
        return func

    # annotate the ir with annotate_target pass
    model = Model()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    y = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x, y).mod
    mod = AnnotateTarget([target])(mod)
    expected_func = expected()
    # check the structure of the expected ir and generated ir
    assert tvm.ir.structural_equal(mod["main"], expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
