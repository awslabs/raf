#pylint: disable=invalid-name,protected-access
"""Utilities for testing and benchmarks"""
import numpy as np
import torch
import mnm
import tvm

from .._core.module import Module
from .._core.executor import VMExecutor, VMCompiler
from .._ffi import ir
from .._ffi import pass_

def run_infer_type(func):
    """Helper function to infer the type of the given function"""
    main = tvm.relay.GlobalVar("main")
    mod = ir._make.Module({main: func})
    mod = pass_.InferType(mod)
    return mod["main"]


def check_type(expr, typ):
    """Helper function to check expr.checked_type == typ"""
    checked_type = expr.checked_type
    if checked_type != typ:
        raise RuntimeError(f"Type mismatch {checked_type} vs {typ}")


def get_ctx_list():
    """Helper function to get all available contexts"""
    ret = ["llvm"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def check(m_x, m_y, *, rtol=1e-5, atol=1e-5):
    """Helper function to check if m_x and m_y are equal"""
    m_x = m_x.asnumpy()
    m_y = m_y.asnumpy()
    np.testing.assert_allclose(m_x, m_y, rtol=rtol, atol=atol)


def randn(shape, *, ctx="cpu", dtype="float32"):
    """Helper function to generate a pair of mnm and numpy arrays"""
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randint(shape, *, low=0, high=None, ctx="cpu", dtype="int64"):
    """Helper function to generate a pair of mnm and numpy arrays with int"""
    x = np.random.randint(low, high, shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randn_torch(shape, *, ctx="cpu", dtype="float32", requires_grad=True, std=1.0):
    """Helper function to generate a pair of mnm and torch arrays"""
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    t_x = torch.tensor(n_x, requires_grad=requires_grad)  # pylint: disable=not-callable
    return m_x, t_x


def run_vm_model(model, ctx, args):
    """Helper function to execute model with VM"""
    mod = Module()
    func = model._internal(*args).func
    mod[tvm.ir.GlobalVar('main')] = func
    executor = VMExecutor(mod, ctx)
    out = executor.make_executor()(*args)
    return out


def compile_vm_model(model, ctx, args):
    """Helper function to compile model into VM bytecode"""
    mod = Module()
    func = model._internal(*args).func
    mod[tvm.ir.GlobalVar('main')] = func
    executor = VMExecutor(mod, ctx)
    return executor.executable.bytecode


def lower_vm_model(model, target_name, args):
    """Helper function to lower model into optimized relay"""
    mod = Module()
    func = model._internal(*args).func
    gvar = tvm.ir.GlobalVar('main')
    mod[gvar] = func
    compiler = VMCompiler()
    mod, _ = compiler.optimize(mod, target_name)
    return mod[gvar]
