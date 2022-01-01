# pylint: disable=missing-class-docstring,missing-function-docstring
"""Module that consists of global variables and functions."""
import mnm._ffi.ir.module as ffi
from mnm._lib import IRModule  # pylint: disable=unused-import


def get_global():
    return ffi.Global()
