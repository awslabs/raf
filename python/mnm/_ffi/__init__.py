"""MNM is Not MXNet, it's MXNet 2.0.

C interfacing code.

This namespace contains everything that interacts with C code,
most of which are borrowed or directly imported from TVM.
"""
from . import _tvm
from . import libinfo

from . import attrs
from . import op
from . import value
