"""SGD optimizer."""
import numpy as np
from tvm import relay
from mnm._ffi.ir.constant import ExtractValue
from mnm._ffi.binding import LookupBoundExpr
from mnm._core.value import NoGradValue
from .._core.core_utils import get_chained_attr
from .._core.ndarray import ndarray, array, get_symbol_handle
from .optim import with_autodiff
from ..model import trace, Model, trace_mutate_attr
from ..model.trace import _get_func_inputs
from .._op import imp
from .._op.sym import multiply, add, subtract


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


def has_grad(dx):
    """ Check if dx is NoGradValue """
    def simplify(x):
        if isinstance(x, relay.Var):
            return simplify(LookupBoundExpr(x))
        if isinstance(x, relay.TupleGetItem):
            tup = simplify(x.tuple_value)
            if isinstance(tup, relay.Tuple):
                return simplify(tup[x.index])
        return x

    dx = simplify(get_symbol_handle(dx))
    if isinstance(dx, relay.Constant):
        dx = ExtractValue(dx)
        return not isinstance(dx, NoGradValue)
    return True


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
                self.learning_rate = array(learning_rate, dtype='float32')
                self.momentum = array(momentum, dtype='float32')
                self.params = {}
                for name, x in self.model.state().items():
                    if x.requires_grad is True:
                        assert isinstance(x, ndarray), "Only `mnm.ndarray' can be optimized!"
                        npa = np.zeros(x.shape, dtype=x.dtype)
                        v_i = ndarray(npa, device=x.device, name=f'{name}.v')
                        setattr(self, f'{name}.v', v_i)
                        self.params[x._ndarray__handle] = (name, x, v_i)

            @trace
            def forward(self, dy, *args):
                # pylint: disable=missing-function-docstring, invalid-name
                y, dxs = self.ad_model(dy, *args)
                record = self.ad_model._internal(dy, *args)
                inputs = _get_func_inputs(record, args, {})
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, x, v = self.params[param]
                        new_v = add(dxi, multiply(self.momentum, v), out=v)
                        new_x = subtract(x, multiply(self.learning_rate, new_v), out=x)
                        param_model = get_chained_attr(self.model, name.split('.')[:-1])
                        trace_mutate_attr(param_model, name.split('.')[-1], new_x)
                        trace_mutate_attr(self, f'{name}.v', new_v)
                return y
        return SGDWrapper(model)
    return decorator
