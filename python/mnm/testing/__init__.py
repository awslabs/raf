#pylint: disable=invalid-name
"""Utilities for testing and benchmarks"""
import numpy as np
import torch
import mnm
import tvm

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


def randn(shape, *, ctx="cpu", dtype="float32"):
    """Helper function to generate a pair of mnm and numpy arrays"""
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    return m_x, n_x


def randn_torch(shape, *, ctx="cpu", dtype="float32", std=1.0):
    """Helper function to generate a pair of mnm and torch arrays"""
    x = np.random.randn(*shape) * std
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    t_x = torch.tensor(n_x, requires_grad=True)  # pylint: disable=not-callable
    return m_x, t_x
