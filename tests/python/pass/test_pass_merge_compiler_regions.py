# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access
import pytest
import numpy as np
import raf
from raf._ffi.pass_ import AnnotateTarget, MergeCompilerRegions
from raf._core.ir_ext import extended_var
from raf._lib import tvm
from raf._lib import relay as _relay


def make_compiler_attrs(compiler):
    return tvm.ir.make_node("relay.attrs.CompilerAttrs", compiler=compiler)


def test_single_input_output_merge():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    target = "test_single_input_output_merge"

    @tvm.ir.register_op_attr("raf.op.relu", "target." + target)
    def relu(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.copy", "target." + target)
    def copy(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.negative", "target." + target)
    def negative(attrs, args):  # pylint: disable=unused-argument
        return True

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            a_1 = raf.relu(x)
            a_2 = raf.abs(a_1)
            a_3 = raf.copy(a_2)
            out = raf.negative(a_3)
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = raf.op.compiler_begin(%x, meta[relay.attrs.CompilerAttrs][0]);
        # %1 = raf.op.relu(%0);
        # let %a1 = raf.op.compiler_end(%1, meta[relay.attrs.CompilerAttrs][1]);
        # %2 = raf.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][2]);
        # %3 = raf.op.abs(%2);
        # let %a2 = raf.op.compiler_end(%3, meta[relay.attrs.CompilerAttrs][3]);
        # %4 = raf.op.compiler_begin(%a2, meta[relay.attrs.CompilerAttrs][4]);
        # let %a3 = raf.op.copy(%4);
        # %5 = raf.op.negative(%a3, -114514, -114514);
        # let %a4 = raf.op.compiler_end(%5, meta[relay.attrs.CompilerAttrs][5]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        # define calls
        relu_call = raf.ir.op.compiler_begin(x, attrs=make_compiler_attrs(target))
        relu_call = raf.ir.op.relu(relu_call)
        relu_call = raf.ir.op.compiler_end(relu_call, attrs=make_compiler_attrs(target))
        abs_call = raf.ir.op.compiler_begin(a1, attrs=make_compiler_attrs("default"))
        abs_call = raf.ir.op.abs(abs_call)
        abs_call = raf.ir.op.compiler_end(abs_call, attrs=make_compiler_attrs("default"))
        copy_call = raf.ir.op.compiler_begin(a2, attrs=make_compiler_attrs(target))
        copy_call = raf.ir.op.copy(copy_call)
        negative_call = raf.ir.op.negative(a3)
        negative_call = raf.ir.op.compiler_end(negative_call, make_compiler_attrs(target))
        # make anf
        body = _relay.Let(a4, negative_call, a4)
        body = _relay.Let(a3, copy_call, body)
        body = _relay.Let(a2, abs_call, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate ir and merge compiler regions
    model = Model()
    x = raf.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)["main"]
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

    @tvm.ir.register_op_attr("raf.op.relu", "target." + target)
    def relu(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.abs", "target." + target)
    def abs(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.add", "target." + target)
    def add(attrs, args):  # pylint: disable=unused-argument
        return True

    class MergeableModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            r = raf.relu(x)
            a_1 = raf.abs(r)
            a_2 = raf.tanh(r)
            out = raf.add(a_1, a_2)
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = raf.op.compiler_begin(%x, meta[relay.attrs.CompilerAttrs][0]);
        # let %a1 = raf.op.relu(%0);
        # %1 = raf.op.abs(%a1);
        # let %a2 = raf.op.compiler_end(%1, meta[relay.attrs.CompilerAttrs][1]);
        # %2 = raf.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][2]);
        # %3 = raf.op.tanh(%2);
        # let %a3 = raf.op.compiler_end(%3, meta[relay.attrs.CompilerAttrs][3]);
        # %4 = raf.op.compiler_begin(%a2, meta[relay.attrs.CompilerAttrs][4]);
        # %5 = raf.op.compiler_begin(%a3, meta[relay.attrs.CompilerAttrs][5]);
        # %6 = raf.op.compiler_begin(bool(0), meta[relay.attrs.CompilerAttrs][6]);
        # %7 = raf.op.add(%4, %5, %6);
        # let %a4 = raf.op.compiler_end(%7, meta[relay.attrs.CompilerAttrs][7]);
        # %a4
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        # define calls
        relu_call = raf.ir.op.compiler_begin(x, attrs=make_compiler_attrs(target))
        relu_call = raf.ir.op.relu(relu_call)
        abs_call = raf.ir.op.abs(a1)
        abs_call = raf.ir.op.compiler_end(abs_call, attrs=make_compiler_attrs(target))
        tanh_call = raf.ir.op.compiler_begin(a1, attrs=make_compiler_attrs("default"))
        tanh_call = raf.ir.op.tanh(tanh_call)
        tanh_call = raf.ir.op.compiler_end(tanh_call, make_compiler_attrs("default"))
        add_call1 = raf.ir.op.compiler_begin(a2, attrs=make_compiler_attrs(target))
        add_call2 = raf.ir.op.compiler_begin(a3, attrs=make_compiler_attrs(target))
        const_call1 = raf.ir.op.compiler_begin(None, attrs=make_compiler_attrs(target))
        const_call2 = raf.ir.op.compiler_begin(None, attrs=make_compiler_attrs(target))
        add_call = raf.ir.op.add(add_call1, add_call2, const_call1, const_call2)
        add_call = raf.ir.op.compiler_end(add_call, attrs=make_compiler_attrs(target))
        # make anf
        body = _relay.Let(a4, add_call, a4)
        body = _relay.Let(a3, tanh_call, body)
        body = _relay.Let(a2, abs_call, body)
        body = _relay.Let(a1, relu_call, body)
        func = _relay.Function([x], body)
        return func

    # annotate ir and merge compiler regions
    model = MergeableModel()
    x = raf.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)["main"]
    expected_func = expected()
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


def test_tuple_merge():
    # pylint: disable=no-self-use, redefined-builtin, too-many-locals, invalid-name, unused-variable
    target = "test_tuple_merge"

    @tvm.ir.register_op_attr("raf.op.relu", "target." + target)
    def relu(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.tanh", "target." + target)
    def tanh(attrs, args):  # pylint: disable=unused-argument
        return True

    @tvm.ir.register_op_attr("raf.op.concatenate", "target." + target)
    def concatenate(attrs, args):  # pylint: disable=unused-argument
        return True

    class MergeableModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            r = raf.relu(x)
            a_1 = raf.abs(r)
            a_2 = raf.tanh(r)
            out = raf.concatenate((a_1, a_2))
            return out

    def expected():
        # build the expected ir after merge compiler regions
        # expected_func:
        # fn (%x: Tensor[(10, 10), float64]) {
        # %0 = raf.op.compiler_begin(%x, meta[relay.attrs.CompilerAttrs][0]);
        # %1 = raf.op.relu(%0);
        # let %a1 = raf.op.compiler_end(%1, meta[relay.attrs.CompilerAttrs][1]);
        # %2 = raf.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][2]);
        # %3 = raf.op.abs(%2);
        # let %a2 = raf.op.compiler_end(%3, meta[relay.attrs.CompilerAttrs][3]);
        # %4 = raf.op.compiler_begin(%a1, meta[relay.attrs.CompilerAttrs][4]);
        # let %a3 = raf.op.tanh(%4);
        # let %a4 = (%a2, %a3);
        # %5 = raf.op.concatenate(%a4, -114514);
        # let %a5 = raf.op.compiler_end(%5, meta[relay.attrs.CompilerAttrs][5]);
        # %a5
        # }
        # define variables
        x = extended_var("x", dtype="float64", shape=(10, 10))
        a1 = extended_var("a1")
        a2 = extended_var("a2")
        a3 = extended_var("a3")
        a4 = extended_var("a4")
        a5 = extended_var("a5")
        # define calls
        relu_call = raf.ir.op.compiler_begin(x, attrs=make_compiler_attrs(target))
        relu_call = raf.ir.op.relu(relu_call)
        relu_call = raf.ir.op.compiler_end(relu_call, attrs=make_compiler_attrs(target))
        abs_call = raf.ir.op.compiler_begin(a1, attrs=make_compiler_attrs("default"))
        abs_call = raf.ir.op.abs(abs_call)
        abs_call = raf.ir.op.compiler_end(abs_call, attrs=make_compiler_attrs("default"))
        tanh_call = raf.ir.op.compiler_begin(a1, attrs=make_compiler_attrs(target))
        tanh_call = raf.ir.op.tanh(tanh_call)
        concat_tuple = _relay.Tuple([a2, a3])
        concat_call = raf.ir.op.concatenate(a4)
        concat_call = raf.ir.op.compiler_end(concat_call, attrs=make_compiler_attrs(target))
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
    x = raf.array(np.random.randn(10, 10), dtype="float64")
    mod = model._internal(x).mod
    mod = AnnotateTarget([target])(mod)
    func = MergeCompilerRegions()(mod)["main"]
    expected_func = expected()
    # check ir structure
    assert tvm.ir.structural_equal(func, expected_func)


if __name__ == "__main__":
    pytest.main([__file__])
