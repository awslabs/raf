"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.dev"

import readline as _

from . import _base
from . import _ffi
from . import _context

from ._context import cpu, gpu
from ._ndarray import ndarray
from .hybrid import hybrid
