"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.dev"

import readline

from . import _ffi
from . import _core
from .hybrid import hybrid

from . import op
from .op import nn

array = op.imports.array

from .op.arith import *
