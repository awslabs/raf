# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, attribute-defined-outside-init, too-many-locals
# pylint: disable=too-many-statements, no-self-use, too-many-arguments
import pytest
import raf
from raf.ir import RAFSequential, ScopeBuilder
from raf.testing import run_infer_type
from raf._ffi.pass_ import InferType, FullInline

import tvm
from tvm import relay


def check_inlined_ir(orig, golden, use_structural_equal=True):
    """
    Check if inlining actually does what we want. Only checks if the two IRs
    are structurally equal.
    """
    passes = [InferType(), FullInline(), InferType()]
    pass_seq = RAFSequential(passes)
    inlined = pass_seq(orig)
    golden = run_infer_type(golden)
    if use_structural_equal:
        assert tvm.ir.structural_equal(inlined, golden), "\nExpected:\n%s\nGot\n%s" % (
            raf.ir.AsText(golden),
            raf.ir.AsText(inlined),
        )
    else:
        assert raf.ir.AsText(golden) == raf.ir.AsText(inlined), "\nExpected:\n%s\nGot\n%s" % (
            raf.ir.AsText(golden),
            raf.ir.AsText(inlined),
        )


# Simple test to see if inlining works as desired
def test_simple_inline():
    shape = (16, 16)
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def get_mod_multi_func():
        """
        Toy example with additions inside function bodies.

        fn(inp0, inp1, inp2, inp3) {

            // Function to be inlined
            let foo = fn(x, y) {
                let x_1 = add(x, y)
                let x_2 = add(x_1, y)
                x_2
            }

            // Call it three times
            let a0 = foo(inp0, inp1)
            let a1 = foo(inp2, inp3)
            let a2 = foo(a0, a1)
            a2
        }
        """
        inp0 = raf.ir.var("inp0", shape=shape)
        inp1 = raf.ir.var("inp1", shape=shape)
        inp2 = raf.ir.var("inp2", shape=shape)
        inp3 = raf.ir.var("inp3", shape=shape)

        sb = ScopeBuilder()
        # Function
        x = raf.ir.var("x", shape=shape)
        y = raf.ir.var("y", shape=shape)
        sb_foo = ScopeBuilder()
        x_1 = sb_foo.let("x_1", relay.Call(add_op, [x, y, null, null]))
        x_2 = sb_foo.let("x_2", relay.Call(add_op, [x_1, y, null, null]))
        sb_foo.ret(x_2)
        func_foo = relay.Function([x, y], sb_foo.get())
        foo_var = sb.let("foo", func_foo)

        # Call the function
        a_0 = sb.let("a0", relay.Call(foo_var, [inp0, inp1]))
        a_1 = sb.let("a1", relay.Call(foo_var, [inp2, inp3]))
        a_2 = sb.let("a2", relay.Call(foo_var, [a_0, a_1]))
        sb.ret(a_2)
        func = relay.Function([inp0, inp1, inp2, inp3], sb.get())
        return tvm.IRModule.from_expr(func)

    def get_mod_inlined():
        """
        The inlined version of the toy example above. Notice that we don't eliminate the
        let vars for the calls. We just assign them with different RHS, which will result in
        a direct assign. However, DCE will remove it afterwards.

        fn(inp0, inp1, inp2, inp3) {
            let x_1_0 = add(inp0, inp1)
            let x_2_0 = add(x_1_0, inp1)
            let x_1_1 = add(inp2, inp3)
            let x_2_1 = add(x_1_1, inp3)
            let x_1_2 = add(x_2_0, x_2_1)
            let x_2_2 = add(x_1_2, x_2_1)
            x_2_2
        }

        """
        inp0 = raf.ir.var("inp0", shape=shape)
        inp1 = raf.ir.var("inp1", shape=shape)
        inp2 = raf.ir.var("inp2", shape=shape)
        inp3 = raf.ir.var("inp3", shape=shape)
        sb = ScopeBuilder()

        x_1_0 = sb.let("x_1_0", relay.Call(add_op, [inp0, inp1, null, null]))
        x_2_0 = sb.let("x_2_0", relay.Call(add_op, [x_1_0, inp1, null, null]))
        x_1_1 = sb.let("x_1_1", relay.Call(add_op, [inp2, inp3, null, null]))
        x_2_1 = sb.let("x_2_1", relay.Call(add_op, [x_1_1, inp3, null, null]))
        x_1_2 = sb.let("x_1_2", relay.Call(add_op, [x_2_0, x_2_1, null, null]))
        x_2_2 = sb.let("x_2_2", relay.Call(add_op, [x_1_2, x_2_1, null, null]))
        sb.ret(x_2_2)
        func = relay.Function([inp0, inp1, inp2, inp3], sb.get())
        return tvm.IRModule.from_expr(func)

    check_inlined_ir(get_mod_multi_func(), get_mod_inlined())


# See if inlining can handle more than one level of function calls
def test_multi_level():
    shape = (16, 16)
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def get_mod_multi_func():
        """
        Nested function calls.

        // Inner-level function
        foo_inner(x, y) {
            let x_1 = add(x, y)
            let x_2 = add(x_1, y)
            x_2
        }

        // Outer-level function
        foo_outer(a, b) {
            let a_1 = foo_inner(a, b)
            let a_2 = add(a_1, b)
            a_2
        }

        main(inp0, inp1, inp2, inp3) {
            // Call the functions
            let v0 = foo_inner(inp0, inp1)
            let v1 = foo_outer(inp2, inp3)
            let v2 = add(v0, v1)
            v2
        }
        """
        inp0 = raf.ir.var("inp0", shape=shape)
        inp1 = raf.ir.var("inp1", shape=shape)
        inp2 = raf.ir.var("inp2", shape=shape)
        inp3 = raf.ir.var("inp3", shape=shape)

        mod = tvm.IRModule()
        # Inner-level function
        x = raf.ir.var("x", shape=shape)
        y = raf.ir.var("y", shape=shape)
        sb_inner = ScopeBuilder()
        x_1 = sb_inner.let("x_1", relay.Call(add_op, [x, y, null, null]))
        x_2 = sb_inner.let("x_2", relay.Call(add_op, [x_1, y, null, null]))
        sb_inner.ret(x_2)
        func_foo_inner = relay.Function([x, y], sb_inner.get())
        foo_inner = tvm.ir.GlobalVar("foo_inner")
        mod.update_func(foo_inner, func_foo_inner)

        # Outer-level function
        a = raf.ir.var("a", shape=shape)
        b = raf.ir.var("b", shape=shape)
        sb_outer = ScopeBuilder()
        a_1 = sb_outer.let("a_1", relay.Call(foo_inner, [a, b]))
        a_2 = sb_outer.let("a_2", relay.Call(add_op, [a_1, b, null, null]))
        sb_outer.ret(a_2)
        func_foo_outer = relay.Function([a, b], sb_outer.get())
        foo_outer = tvm.ir.GlobalVar("foo_outer")
        mod.update_func(foo_outer, func_foo_outer)

        # Call the functions
        sb = ScopeBuilder()
        v_0 = sb.let("v0", relay.Call(foo_inner, [inp0, inp1]))
        v_1 = sb.let("v1", relay.Call(foo_outer, [inp2, inp3]))
        v_2 = sb.let("v2", relay.Call(add_op, [v_0, v_1, null, null]))
        sb.ret(v_2)
        func = relay.Function([inp0, inp1, inp2, inp3], sb.get())
        main = tvm.ir.GlobalVar("main")
        mod.update_func(main, func)
        return mod

    def get_mod_inlined():
        """
        fn(inp0, inp1, inp2, inp3) {
            let x_1_inner_0 = add(inp0, inp1)
            let x_2_inner_0 = add(x_1_inner_0, inp1)
            let x_1_inner_1 = add(inp2, inp3)
            let x_2_inner_1 = add(x_1_inner_1, inp3)
            let a_2_outer = add(x_2_inner_1, inp3)
            let a2 = add(x_2_inner_0, a_2_outer)
            a2
        }

        """
        inp0 = raf.ir.var("inp0", shape=shape)
        inp1 = raf.ir.var("inp1", shape=shape)
        inp2 = raf.ir.var("inp2", shape=shape)
        inp3 = raf.ir.var("inp3", shape=shape)
        sb = ScopeBuilder()

        x_1_inner_0 = sb.let("x_1_inner_0", relay.Call(add_op, [inp0, inp1, null, null]))
        x_2_inner_0 = sb.let("x_2_inner_0", relay.Call(add_op, [x_1_inner_0, inp1, null, null]))
        x_1_inner_1 = sb.let("x_1_inner_1", relay.Call(add_op, [inp2, inp3, null, null]))
        x_2_inner_1 = sb.let("x_2_inner_1", relay.Call(add_op, [x_1_inner_1, inp3, null, null]))
        a_2_outer = sb.let("a_2_outer", relay.Call(add_op, [x_2_inner_1, inp3, null, null]))
        a_2 = sb.let("a2", relay.Call(add_op, [x_2_inner_0, a_2_outer, null, null]))
        sb.ret(a_2)
        func = relay.Function([inp0, inp1, inp2, inp3], sb.get())
        return tvm.IRModule.from_expr(func)

    check_inlined_ir(get_mod_multi_func(), get_mod_inlined())


# See if the inlining pass can detect cycles in the call graph. We skip the entire pass
# in this case
def test_recursion():
    shape = (16, 16)
    add_op = raf._ffi.op.GetOp("raf.op.add")
    null = raf.ir.const(None)

    def get_mod_multi_func():
        """
        Call graph with a cycle

        foo0(x, y) {
            let x_1 = foo1(x, y)
            let x_2 = add(x_1, y)
            x_2
        }

        foo1(a, b) {
            let a_1 = foo0(a, b)
            let a_2 = add(a_1, b)
            a_2
        }

        main(inp0, inp1, inp2) {
            let v0 = foo1(inp0, inp1)
            let v1 = add(v0, inp2)
            v1
        }
        """
        inp0 = raf.ir.var("inp0", shape=shape)
        inp1 = raf.ir.var("inp1", shape=shape)
        inp2 = raf.ir.var("inp2", shape=shape)

        mod = tvm.IRModule()
        foo0 = tvm.ir.GlobalVar("foo0")
        foo1 = tvm.ir.GlobalVar("foo1")

        # First function
        x = raf.ir.var("x", shape=shape)
        y = raf.ir.var("y", shape=shape)
        sb0 = ScopeBuilder()
        x_1 = sb0.let("x_1", relay.Call(foo1, [x, y]))
        x_2 = sb0.let("x_2", relay.Call(add_op, [x_1, y, null, null]))
        sb0.ret(x_2)
        func_foo0 = relay.Function([x, y], sb0.get(), ret_type=relay.TensorType(shape))
        mod.update_func(foo0, func_foo0)

        # Second function
        a = raf.ir.var("a", shape=shape)
        b = raf.ir.var("b", shape=shape)
        sb1 = ScopeBuilder()
        a_1 = sb1.let("a_1", relay.Call(foo0, [a, b]))
        a_2 = sb1.let("a_2", relay.Call(add_op, [a_1, b, null, null]))
        sb1.ret(a_2)
        func_foo1 = relay.Function([a, b], sb1.get(), ret_type=relay.TensorType(shape))
        mod.update_func(foo1, func_foo1)

        # Main function
        sb = ScopeBuilder()
        v_0 = sb.let("a0", relay.Call(foo1, [inp0, inp1]))
        v_1 = sb.let("a1", relay.Call(add_op, [v_0, inp2, null, null]))
        sb.ret(v_1)
        func = relay.Function([inp0, inp1, inp2], sb.get())
        main = tvm.ir.GlobalVar("main")
        mod.update_func(main, func)
        return mod

    # The IR should not change, but structural_equal does not seem to work on
    # recursive functions. So we check the text version instead.
    check_inlined_ir(
        get_mod_multi_func(),
        get_mod_multi_func(),
        use_structural_equal=False,
    )


if __name__ == "__main__":
    pytest.main([__file__])
