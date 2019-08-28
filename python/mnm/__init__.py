"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.dev"

import readline

from . import _ffi
from . import _core
from .hybrid import hybrid

from . import op

array = op.imports.array

add = op.arithmetic.add
subtract = op.arithmetic.subtract
multiply = op.arithmetic.multiply
divide = op.arithmetic.divide
mod = op.arithmetic.mod
negative = op.arithmetic.negative
