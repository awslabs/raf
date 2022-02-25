# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for relay pass manager."""
# pylint: disable=missing-function-docstring, invalid-name, missing-class-docstring
# pylint: disable=too-few-public-methods, unused-argument

import pytest

import tvm
from tvm import relay
from tvm.ir.transform import ModulePass, PassContext
from tvm.ir.transform import module_pass
from tvm.relay.transform import function_pass, FunctionPass
from raf._ffi import pass_
from raf._ffi.pass_ import FromRelay
from raf.ir import RAFSequential


def get_var_func():
    shape = (5, 10)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("myAbs")
    func = relay.Function([x], relay.abs(x))
    return gv, func


def extract_var_func(mod, name):
    var = mod.get_global_var(name)
    func = mod[var]
    return var, func


def update_func(func):
    # Double the value of Constants and vars.
    class DoubleValues(relay.ExprMutator):
        def __init__(self):
            relay.ExprMutator.__init__(self)

        def visit_constant(self, const):
            return relay.add(const, const)

        def visit_var(self, var):
            return relay.add(var, var)

        def visit_call(self, call):
            new_op = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            return relay.Call(new_op, new_args, call.attrs)

        def visit_global_var(self, gvar):
            return gvar

        def visit_op(self, op):
            return op

        def visit_function(self, fn):
            new_body = self.visit(fn.body)
            return relay.Function(list(fn.params), new_body, fn.ret_type, fn.type_params, fn.attrs)

    double_value = DoubleValues()
    return double_value.visit(func)


class OptTester:
    """A helper class for testing the pass manager."""

    def __init__(self, mod):
        if not isinstance(mod, tvm.IRModule):
            raise TypeError(
                "mod is expected to be the type of \
                            tvm.IRModule"
            )
        self.mod = mod

    @staticmethod
    def transform(node, dev=None):
        """Perform optimization on node."""
        if isinstance(node, tvm.IRModule):
            # Add a function to the module and return an updated module.
            gv, func = get_var_func()
            mod = tvm.IRModule({gv: func})
            mod.update(node)
            return mod
        if isinstance(node, relay.Function):
            return update_func(node)

        raise TypeError("Found not supported node type.")


def test_module_pass():
    shape = (5, 10)
    dtype = "float32"
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    v_add = relay.GlobalVar("myAdd")
    func = relay.Function([x, y], x + y)
    mod = tvm.IRModule({v_add: func})

    opt_tester = OptTester(mod)

    def direct_transform(expr, dev):
        return opt_tester.transform(expr, dev)

    mod_pass = module_pass(direct_transform, opt_level=3, name="module_pass_test")
    assert isinstance(mod_pass, ModulePass)
    pass_info = mod_pass.info
    assert pass_info.name == "module_pass_test"
    assert pass_info.opt_level == 3


def test_function_pass():
    shape = (10,)
    dtype = "float32"
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    v_log = relay.GlobalVar("myLog")
    log = relay.Function([x], relay.log(x))
    mod = tvm.IRModule({v_log: log})

    pass_name = "function_pass_test"
    opt_tester = OptTester(mod)

    def direct_transform(expr, dev):
        return opt_tester.transform(expr, dev)

    func_pass = function_pass(direct_transform, opt_level=1, name=pass_name)
    assert isinstance(func_pass, FunctionPass)
    pass_info = func_pass.info
    assert pass_info.name == "function_pass_test"
    assert pass_info.opt_level == 1


def test_sequential_pass():
    shape = (10,)
    dtype = "float32"
    tp = relay.TensorType(shape, dtype)
    x = relay.var("x", tp)
    y = relay.var("y", tp)
    v_sub = relay.GlobalVar("mySub")
    sub = relay.Function([x, y], relay.subtract(x, y))

    z = relay.var("z", tp)
    v_log = relay.GlobalVar("myLog")
    log = relay.Function([z], relay.log(z))

    mod = tvm.IRModule({v_sub: sub, v_log: log})
    tvm_mod = FromRelay()(mod)
    passes = [pass_.InferType(), pass_.InferType()]
    sequential = RAFSequential(passes=passes, opt_level=1, name="seq")
    with PassContext():
        ret_mod = sequential(tvm_mod)

    assert isinstance(ret_mod["mySub"].body.checked_type, tvm.ir.TensorType)


if __name__ == "__main__":
    pytest.main([__file__])
