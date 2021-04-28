# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm._ffi.pass_ import AnnotateTarget, MergeCompilerRegions
from mnm._core.ir_ext import extended_var
from mnm._lib import tvm
from mnm._lib import relay as _relay


def test_single_input_output_merge():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    target = "test_single_input_output_merge"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.copy", "target." + target)
    def copy(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.negative", "target." + target)
    def negative(attrs, args): # pylint: disable=unused-argument
        return True

    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            a_1 = mnm.relu(x)
            a_2 = mnm.abs(a_1)
            a_3 = mnm.copy(a_2)
            out = mnm.negative(a_3)
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = mnm.op.compiler_begin(%x, meta[mnm.args.compiler][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[mnm.args.compiler][1]);
        # %2 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][2]);
        # %3 = mnm.op.abs(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[mnm.args.compiler][3]);
        # %4 = mnm.op.compiler_begin(%a2, meta[mnm.args.compiler][4]);
        # let %a3 = mnm.op.copy(%4);
        # %5 = mnm.op.negative(%a3, -114514, -114514);
        # let %a4 = mnm.op.compiler_end(%5, meta[mnm.args.compiler][5]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        const = _relay.Constant(tvm.nd.array(-114514))
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        copy = _relay.op.get("mnm.op.copy")
        negative = _relay.op.get("mnm.op.negative")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call = _relay.Call(begin, [x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call = _relay.Call(relu, [relu_call])
        relu_call = _relay.Call(end, [relu_call], tvm.ir.make_node("mnm.args.compiler"))
        abs_call = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        abs_call = _relay.Call(abs, [abs_call])
        abs_call = _relay.Call(end, [abs_call], tvm.ir.make_node("mnm.args.compiler"))
        copy_call = _relay.Call(begin, [a2], tvm.ir.make_node("mnm.args.compiler"))
        copy_call = _relay.Call(copy, [copy_call])
        negative_call = _relay.Call(negative, [a3])
        negative_call = _relay.Call(end, [negative_call], tvm.ir.make_node("mnm.args.compiler"))
        # make anf
        body = _relay.Let(a4, negative_call, a4)
        body = _relay.Let(a3, copy_call, body)
        body = _relay.Let(a2, abs_call, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate ir and merge compiler regions
    model = Model()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)['main']
    expected_func = expected()
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


def test_diamond_merge():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    """
    This tests that the data dependencies present in a diamond-shaped
    graph are correctly resolved by the merging pass.

    O = supported by target
    X = not supported by target

       O         O
      / \\      /               \\
     O   X --> O    +       +    X
     \\ /             \\ /
       O                O

    Note that we can't just merge the three supported operators together,
    otherwise both subgraphs would depend on the other (a.k.a dead lock).
    """

    target = "test_diamond_merge"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.abs", "target." + target)
    def abs(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.add", "target." + target)
    def add(attrs, args): # pylint: disable=unused-argument
        return True

    class MergeableModel(mnm.Model):
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
        # %0 = mnm.op.compiler_begin(%x, meta[mnm.args.compiler][0]);
        # let %a1 = mnm.op.relu(%0);
        # %1 = mnm.op.abs(%a1);
        # let %a2 = mnm.op.compiler_end(%1, meta[mnm.args.compiler][1]);
        # %2 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][2]);
        # %3 = mnm.op.tanh(%2);
        # let %a3 = mnm.op.compiler_end(%3, meta[mnm.args.compiler][3]);
        # %4 = mnm.op.compiler_begin(%a2, meta[mnm.args.compiler][4]);
        # %5 = mnm.op.compiler_begin(%a3, meta[mnm.args.compiler][5]);
        # %6 = mnm.op.compiler_begin(-114514, meta[mnm.args.compiler][6]);
        # %7 = mnm.op.compiler_begin(-114514, meta[mnm.args.compiler][7]);
        # %8 = mnm.op.add(%4, %5, %6, %7);
        # let %a4 = mnm.op.compiler_end(%8, meta[mnm.args.compiler][8]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        tanh = _relay.op.get("mnm.op.tanh")
        add = _relay.op.get("mnm.op.add")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call = _relay.Call(begin, [x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call = _relay.Call(relu, [relu_call])
        abs_call = _relay.Call(abs, [a1])
        abs_call = _relay.Call(end, [abs_call], tvm.ir.make_node("mnm.args.compiler"))
        tanh_call = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        tanh_call = _relay.Call(tanh, [tanh_call])
        tanh_call = _relay.Call(end, [tanh_call], tvm.ir.make_node("mnm.args.compiler"))
        add_call1 = _relay.Call(begin, [a2], tvm.ir.make_node("mnm.args.compiler"))
        add_call2 = _relay.Call(begin, [a3], tvm.ir.make_node("mnm.args.compiler"))
        add_call = _relay.Call(add, [add_call1, add_call2])
        add_call = _relay.Call(end, [add_call], tvm.ir.make_node("mnm.args.compiler"))
        # make anf
        body = _relay.Let(a4, add_call, a4)
        body = _relay.Let(a3, tanh_call, body)
        body = _relay.Let(a2, abs_call, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate ir and merge compiler regions
    model = MergeableModel()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)['main']
    expected_func = expected()
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


def test_tuple_merge():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    target = "test_tuple_merge"

    @tvm.ir.register_op_attr("mnm.op.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.tanh", "target." + target)
    def tanh(attrs, args): # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("mnm.op.concatenate", "target." + target)
    def concatenate(attrs, args): # pylint: disable=unused-argument
        return True

    class MergeableModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            r = mnm.relu(x)
            a_1 = mnm.abs(r)
            a_2 = mnm.tanh(r)
            out = mnm.concatenate((a_1, a_2))
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = mnm.op.compiler_begin(%x, meta[mnm.args.compiler][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[mnm.args.compiler][1]);
        # %2 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][2]);
        # %3 = mnm.op.abs(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[mnm.args.compiler][3]);
        # %4 = mnm.op.compiler_begin(%a1, meta[mnm.args.compiler][4]);
        # let %a3 = mnm.op.tanh(%4);
        # let %a4 = (%a2, %a3);
        # %5 = mnm.op.concatenate(%a4, -114514);
        # let %a5 = mnm.op.compiler_end(%5, meta[mnm.args.compiler][5]);
        # %a5
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        a5 = extended_var("a5")
        const = _relay.Constant(tvm.nd.array(-114514))
        relu = _relay.op.get("mnm.op.relu")
        abs = _relay.op.get("mnm.op.abs")
        tanh = _relay.op.get("mnm.op.tanh")
        concatenate = _relay.op.get("mnm.op.concatenate")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call = _relay.Call(begin, [x], tvm.ir.make_node("mnm.args.compiler"))
        relu_call = _relay.Call(relu, [relu_call])
        relu_call = _relay.Call(end, [relu_call], tvm.ir.make_node("mnm.args.compiler"))
        abs_call = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        abs_call = _relay.Call(abs, [abs_call])
        abs_call = _relay.Call(end, [abs_call], tvm.ir.make_node("mnm.args.compiler"))
        tanh_call = _relay.Call(begin, [a1], tvm.ir.make_node("mnm.args.compiler"))
        tanh_call = _relay.Call(tanh, [tanh_call])
        concat_tuple = _relay.Tuple([a2, a3])
        concat_call = _relay.Call(concatenate, [a4, const])
        concat_call = _relay.Call(end, [concat_call], tvm.ir.make_node("mnm.args.compiler"))
        # make anf
        body = _relay.Let(a5, concat_call, a5)
        body = _relay.Let(a4, concat_tuple, body)
        body = _relay.Let(a3, tanh_call, body)
        body = _relay.Let(a2, abs_call, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate ir and merge compiler regions
    model = MergeableModel()
    x = mnm.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)['main']
    expected_func = expected()
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
