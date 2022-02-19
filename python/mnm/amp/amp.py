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

"""Functions for enabling AMP (automatic mixed precision)."""
# pylint: disable=protected-access
from mnm._ffi.pass_ import AutoCast, InferType
from mnm._lib import relay
from mnm.frontend.model import FrameworkModel


def autocast(model, args=None):
    """Convert a model running in single precison to half precision.

    Should run the origin model or run function `get_relay_func` once before.

    Parameters
    ----------
    model : mnm.model.Model
        Origin model running in single precison mode.

    args: Optional[List[mnm.ndarray]]
        The input data of the model.
    """
    args = args if args is not None else []
    mod = model._internal(*args).mod
    mod = AutoCast()(mod)
    mod = InferType()(mod)
    return FrameworkModel(mod, mod, model.state(), dict())


class CustomTypeHint:
    """Temporarily changes the cast type hint(s) of operator(s)."""

    def __init__(self, op_map):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_map : Dict[str, Function]
            The map from op names to cast type hint functions.

        Examples
        --------
        .. code-block:: python

        # Temporarily update FMNMCastRule to a user-defined packed function.
        # After the test is finished, the attr value will be set back to the original value.

        with CustomCastRule({"mnm.op.add": custom_add_type_hint}):
            model = mnm.amp.autocast(model)

        """
        self.op_map = dict()
        self.old_map = dict()
        for op_name, func in op_map.items():
            self.op_map[relay.op.get(op_name)] = func

    def __enter__(self):
        for op, func in self.op_map.items():
            self.old_map[op] = op.get_attr("FMNMCastRule")
            op.reset_attr("FMNMCastRule")
            op.set_attr("FMNMCastRule", func)
        return self

    def __exit__(self, ptype, value, trace):
        for op, func in self.old_map.items():
            op.reset_attr("FMNMCastRule")
            if func:
                op.set_attr("FMNMCastRule", func)
