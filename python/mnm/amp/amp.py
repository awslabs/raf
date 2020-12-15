"""Functions for enabling AMP (automatic mixed precision)."""
# pylint: disable=protected-access

from mnm._ffi.pass_ import AutoCast
from mnm._lib import relay
from mnm.frontend.model import FrameworkModel


def autocast(model):
    """Convert a model running in single precison to half precision.

    Should run the origin model or run function `get_relay_func` once before.

    Parameters
    ----------
    model : mnm.model.Model
        Origin model running in single precison mode.
    """
    assert model._Cacher__cache.keys() != 0, \
           "Please run model or run function `get_relay_func` once."
    func = model._internal().func
    func = AutoCast(func)
    return FrameworkModel(func, func, model.state(), dict())


class CustomCastRule:
    """ Temporarily changes the cast rule(s) of operator(s). """

    def __init__(self, op_map):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_map : Dict[str, Function]
            The map from op names to cast rule functions.

        Examples
        --------
        .. code-block:: python

        # Temporarily update FMNMCastRule to a user-defined packed function.
        # After the test is finished, the attr value will be set back to the original value.

        with CustomCastRule({"mnm.op.add": custom_add_rule}):
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
