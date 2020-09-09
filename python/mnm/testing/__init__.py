#pylint: disable=invalid-name
"""Utilities for testing and benchmarks"""
from __future__ import absolute_import as _abs
import tvm
from .._ffi import ir
from .._ffi import pass_

def run_infer_type(func):
    """Helper function to infer the type of the given function"""
    main = tvm.relay.GlobalVar("main")
    mod = ir._make.Module({main: func})
    mod = pass_.InferType(mod)
    return mod["main"]
