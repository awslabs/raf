# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""MNM is Not MXNet, it's MXNet 3.0."""

__version__ = "0.0.2.dev"

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
