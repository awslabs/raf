"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.dev"

import readline as _

from . import base as _base
from . import _ffi
from .context import cpu, gpu
from .module import Module
from . import value

from .hybrid import hybrid
