"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.2.dev"

from ._core.numpy_bridge import array
from ._op.imp import *
from .hybrid import hybrid
