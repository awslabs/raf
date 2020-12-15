"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.2.dev"

from ._core.ndarray import array, ndarray
from ._op.imp import *  # pylint: disable=redefined-builtin
from . import frontend
from . import amp
from . import random
from . import build
from . import model
from . import _tvmjit
from . import optim
from . import utils
from .model.model import Model
from .hybrid import hybrid
from .distributed import *
