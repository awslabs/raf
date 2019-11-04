"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.2.dev"

from . import _lib
from . import _ffi
from . import _core
from .hybrid import hybrid
from .op.imports import array
