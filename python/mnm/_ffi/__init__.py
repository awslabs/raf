"""This module is not user-facing.

Protocol:

An API registered as

        mnm.$PREFIX.$MODULE.$NAME

is imported to Python module

        mnm._ffi.$PREFIX._$MODULE   (if $MODULE does not start with _)
        mnm._ffi.$PREFIX.$MODULE    (if $MODULE starts with _)

Stubs are contained in the module as well,
which means we do not use standalone pyi files.
This is because not every editor recognize them properly.
"""
from . import _tvm
from . import libinfo

from . import ir
from . import value
from . import executor
from . import op
from . import tensor
