"""Traced Optimizers"""
from .._core.ndarray import Symbol
from .._core.core_utils import with_signature
from ..model.trace import _get_func_inputs
from ..model import Model, trace
from .._ffi.pass_ import AutoDiff, InlineBackward
from .._ffi.binding import BindSymbol


def with_autodiff(model):
    """create a new model by apply autodiff to the input"""
    class AutoDiffWrapper(Model):
        """AutoDiff model

        Parameters
        ----------
        model: the forward model
        """
        def build(self, model):
            # pylint: disable=attribute-defined-outside-init, missing-function-docstring
            self.model = model

        @trace
        @with_signature(model.forward,
                        lambda this, other: this[:-2] + other)
        def forward(self, dy, *args, **kwargs):
            # pylint: disable=protected-access, missing-function-docstring
            record = self.model._internal(*args, **kwargs)
            func = AutoDiff(record.func)
            func = InlineBackward(func)
            func = Symbol.from_expr(BindSymbol(func, "", None))
            inputs = [Symbol.from_expr(arg) for arg in _get_func_inputs(record, args, kwargs)]
            inputs = inputs + [dy]
            out = func(*inputs)
            y = out[0]
            dxs = out[1]
            return y, dxs

    return AutoDiffWrapper(model)
