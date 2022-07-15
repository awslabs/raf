# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RAF Accelerates deep learning Frameworks."""

try:
    from .version import __version__
    from .version import __full_version__
    from .version import __gitrev__
except:  # pylint: disable=bare-except
    __version__ = "dev"
    __full_version__ = "dev"
    __gitrev__ = "unknown"

# We must import the lib first, because it imports TVM dependency,
# and we need to set TVM_LIBRARY_PATH to make sure TVM loads the RAF
# compatible libraries.
from ._lib import *

from ._core.ndarray import array, ndarray
from ._op.imp import *  # pylint: disable=redefined-builtin
from . import frontend
from . import amp
from . import random
from . import build
from . import ir
from . import model
from . import _tvm_op
from . import optim
from . import utils
from . import _core
from ._core.device import device, cpu, cuda, Device
from .model.model import Model
from .hybrid import hybrid
from .distributed import *
