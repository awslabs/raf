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

less = op.arithmetic.less
greater = op.arithmetic.greater
less_equal = op.arithmetic.less_equal
greater_equal = op.arithmetic.greater_equal
equal = op.arithmetic.equal
not_equal = op.arithmetic.not_equal
