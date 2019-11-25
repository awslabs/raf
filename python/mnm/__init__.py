"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.2.dev"

from ._core.ndarray import array, ndarray, Parameter
from ._core.model import Model
from ._op.imp import *
from .hybrid import hybrid
