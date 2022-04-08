# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functions for enabling AMP (automatic mixed precision)."""
# pylint: disable=protected-access
from raf._ffi.pass_ import AutoCast, InferType
from raf._lib import relay
from raf.frontend.model import FrameworkModel


def autocast(model, args=None):
    """Convert a model running in single precison to half precision.

    Parameters
    ----------
    model : raf.model.Model
        Origin model running in single precison mode.

    args: Optional[List[raf.ndarray]]
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

        # Temporarily update FRAFCastRule to a user-defined packed function.
        # After the test is finished, the attr value will be set back to the original value.

        with CustomCastRule({"raf.op.add": custom_add_type_hint}):
            model = raf.amp.autocast(model)

        """
        self.op_map = dict()
        self.old_map = dict()
        for op_name, func in op_map.items():
            self.op_map[relay.op.get(op_name)] = func

    def __enter__(self):
        for op, func in self.op_map.items():
            self.old_map[op] = op.get_attr("FRAFCastRule")
            op.reset_attr("FRAFCastRule")
            op.set_attr("FRAFCastRule", func)
        return self

    def __exit__(self, ptype, value, trace):
        for op, func in self.old_map.items():
            op.reset_attr("FRAFCastRule")
            if func:
                op.set_attr("FRAFCastRule", func)
