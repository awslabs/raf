"""SGD optimizer."""
import numpy as np
from .._core.core_utils import get_chained_attr, with_signature
from .._core.ndarray import ndarray
from .optim import with_autodiff
from ..model import trace, Model, trace_mutate_attr
from ..model.trace import _get_func_inputs
from .._op import imp, sym


# pylint: disable=too-few-public-methods
class SGD:
    """ Optimizer : stochastic gradient descent

    Parameters:
    -----------
    params: dict_values
        iterable of parameters to optimize or dicts defining parameter groups

    learning_rate: float
        learning rate

    momentum: float (optional)
        momentum factor
    """
    def __init__(self, params, learning_rate, momentum=0):
        if learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        self.params = []
        self._lr = learning_rate
        self._momentum = momentum
        for i, x in enumerate(params):
            assert isinstance(x, ndarray), "Only `mnm.ndarray' can be optimized!"
            npa = np.zeros(x.shape, dtype=x.dtype)
            v_i = ndarray(npa, device=x.device, name=f'sgd.{i}.v')
            self.params.append((x, v_i))

    def step(self):
        """Update the parameters with gradients."""
        for x0, v0 in self.params:
            if x0.grad is None:
                continue
            v1, x1 = imp.sgd(x0, x0.grad, v0, self._lr, self._momentum)
            x0.update(x1)
            v0.update(v1)


def with_sgd(learning_rate=0.1, momentum=0.01):
    """ Optimizer : stochastic gradient descent

    Parameters:
    -----------
    learning_rate: float (optional)
        learning rate

    momentum: float (optional)
        momentum factor

    Returns
    ret : function
        The wrapper which wraps a model with sgd
    """
    def decorator(model):
        class SGDWrapper(Model):
            """sgd wrapper model

            Parameters
            ----------
            model: the forward model
            """
            # pylint: disable=attribute-defined-outside-init, protected-access, too-many-locals
            def build(self, model):
                # pylint: disable=missing-function-docstring
                self.model = model
                self.ad_model = with_autodiff(model)
                self.learning_rate = learning_rate
                self.momentum = momentum
                self.params = {}
                for name, x in self.model.state().items():
                    if x.requires_grad is True:
                        assert isinstance(x, ndarray), "Only `mnm.ndarray' can be optimized!"
                        npa = np.zeros(x.shape, dtype=x.dtype)
                        v_i = ndarray(npa, device=x.device, name=f'{name}.v')
                        setattr(self, f'{name}.v', v_i)
                        self.params[x._ndarray__handle] = (name, x, v_i)

            @trace
            @with_signature(model.forward,
                            lambda this, other: this[:-2] + other)
            def forward(self, dy, *args, **kwargs):
                # pylint: disable=missing-function-docstring, invalid-name
                y, dxs = self.ad_model(dy, *args, **kwargs)
                record = self.ad_model._internal(dy, *args, **kwargs)
                inputs = _get_func_inputs(record, args, kwargs)
                for i, param in enumerate(inputs):
                    if param in self.params:
                        dxi = dxs[i] if len(inputs) > 1 else dxs
                        name, x, v = self.params[param]
                        ret = sym.sgd(x, dxi, v, self.learning_rate, self.momentum)
                        new_v = ret[0]
                        new_x = ret[1]
                        param_model = get_chained_attr(self.model, name.split('.')[:-1])
                        trace_mutate_attr(param_model, name.split('.')[-1], new_x)
                        trace_mutate_attr(self, f'{name}.v', new_v)
                return y
        return SGDWrapper(model)
    return decorator
