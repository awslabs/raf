# pylint: disable=protected-access
import pytest
import numpy as np
import mnm
from mnm._ffi.pass_ import AnnotateTarget
from mnm._core.ir_ext import extended_var
from mnm._lib import tvm
from mnm._lib import relay as _relay


def make_compiler_attrs(compiler):
    return tvm.ir.make_node("relay.attrs.CompilerAttrs", compiler=compiler)


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
        # %0 = mnm.op.compiler_begin(%x, meta[relay.attrs.CompilerAttrs][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[relay.attrs.CompilerAttrs][1]);
        # %2 = mnm.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][2]);
        # %3 = mnm.op.abs(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[relay.attrs.CompilerAttrs][3]);
        # %4 = mnm.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][4]);
        # %5 = mnm.op.abs(%4);
        # let %a3 = mnm.op.compiler_end(%5, meta[relay.attrs.CompilerAttrs][5]);
        # %6 = mnm.op.compiler_begin(%a2, meta[relay.attrs.CompilerAttrs][6]);
        # %7 = mnm.op.compiler_begin(%a3, meta[relay.attrs.CompilerAttrs][7]);
        # %8 = mnm.op.compiler_begin(nullptr, meta[relay.attrs.CompilerAttrs][8]);
        # %9 = mnm.op.compiler_begin(nullptr, meta[relay.attrs.CompilerAttrs][9]);
        # %10 = mnm.op.add(%6, %7, %8, %9);
        # let %a4 = mnm.op.compiler_end(%10, meta[relay.attrs.CompilerAttrs][10]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        # define calls
        relu_call = mnm.ir.op.compiler_begin(x, attrs=make_compiler_attrs("test"))
        relu_call = mnm.ir.op.relu(relu_call)
        relu_call = mnm.ir.op.compiler_end(relu_call, attrs=make_compiler_attrs("test"))
        abs_call1 = mnm.ir.op.compiler_begin(a1, attrs=make_compiler_attrs("default"))
        abs_call1 = mnm.ir.op.abs(abs_call1)
        abs_call1 = mnm.ir.op.compiler_end(abs_call1, attrs=make_compiler_attrs("default"))
        abs_call2 = mnm.ir.op.compiler_begin(a1, attrs=make_compiler_attrs("default"))
        abs_call2 = mnm.ir.op.abs(abs_call2)
        abs_call2 = mnm.ir.op.compiler_end(abs_call2, attrs=make_compiler_attrs("default"))
        let_call2 = mnm.ir.op.compiler_begin(a2, attrs=make_compiler_attrs("default"))
        let_call3 = mnm.ir.op.compiler_begin(a3, attrs=make_compiler_attrs("default"))
        const_call1 = mnm.ir.op.compiler_begin(None, attrs=make_compiler_attrs("default"))
        const_call2 = mnm.ir.op.compiler_begin(None, attrs=make_compiler_attrs("default"))
        add_call = mnm.ir.op.add(let_call2, let_call3, const_call1, const_call2)
        add_call = mnm.ir.op.compiler_end(add_call, attrs=make_compiler_attrs("default"))
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
        # %0 = mnm.op.compiler_begin(%x, meta[relay.attrs.CompilerAttrs][0]);
        # %1 = mnm.op.relu(%0);
        # let %a1 = mnm.op.compiler_end(%1, meta[relay.attrs.CompilerAttrs][1]);
        # %2 = mnm.op.compiler_begin(%y, meta[relay.attrs.CompilerAttrs][2]);
        # %3 = mnm.op.relu(%2);
        # let %a2 = mnm.op.compiler_end(%3, meta[relay.attrs.CompilerAttrs][3]);
        # let %a3 = (%a1, %a2);
        # %4 = mnm.op.compiler_begin(%a3, meta[relay.attrs.CompilerAttrs][4]);
        # %5 = mnm.op.compiler_begin(-114514, meta[relay.attrs.CompilerAttrs][5]);
        # %6 = mnm.op.concatenate(%4, %5);
        # let %a4 = mnm.op.compiler_end(%6, meta[relay.attrs.CompilerAttrs][6]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        y = extended_var("y", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        const = mnm.ir.const(None)
        relu = _relay.op.get("mnm.op.relu")
        concatenate = _relay.op.get("mnm.op.concatenate")
        begin = _relay.op.get("mnm.op.compiler_begin")
        end = _relay.op.get("mnm.op.compiler_end")
        # define calls
        relu_call_x = mnm.ir.op.compiler_begin(x, attrs=make_compiler_attrs(target))
        relu_call_x = mnm.ir.op.relu(relu_call_x)
        relu_call_x = mnm.ir.op.compiler_end(relu_call_x, attrs=make_compiler_attrs(target))
        relu_call_y = mnm.ir.op.compiler_begin(y, attrs=make_compiler_attrs(target))
        relu_call_y = mnm.ir.op.relu(relu_call_y)
        relu_call_y = mnm.ir.op.compiler_end(relu_call_y, attrs=make_compiler_attrs(target))
        relu_tuple1 = mnm.ir.op.compiler_begin(a1, attrs=make_compiler_attrs(target))
        relu_tuple2 = mnm.ir.op.compiler_begin(a2, attrs=make_compiler_attrs(target))
        relu_tuple = _relay.Tuple([relu_tuple1, relu_tuple2])
        relu_tuple = mnm.ir.op.compiler_end(relu_tuple, attrs=make_compiler_attrs(target))
        concat_call1 = mnm.ir.op.compiler_begin(a3, attrs=make_compiler_attrs(target))
        concat_call2 = mnm.ir.op.compiler_begin(const, attrs=make_compiler_attrs(target))
        concat_call = mnm.ir.op.concatenate(concat_call1, concat_call2)
        concat_call = mnm.ir.op.compiler_end(concat_call, attrs=make_compiler_attrs(target))
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
