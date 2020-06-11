import numpy as np
from .._op.imp import sgd
from .._core.ndarray import ndarray


# pylint: disable=too-few-public-methods
class SGD:
    """ Optimizer : stochastic gradient descent

    Parameters:
    -----------
    params: dict_values
        iterable of parameters to optimize or dicts defining parameter groups

    lr: float
        learning rate

    momentum: float (optional)
        momentum factor
    """
    def __init__(self, params, lr, momentum=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        self.params = []
        self._lr = lr
        self._momentum = momentum
        for i, x in enumerate(params):
            assert isinstance(x, ndarray), "Only `mnm.ndarray' can be optimized!"
            npa = np.zeros(x.shape, dtype=x.dtype)
            v_i = ndarray(npa, ctx=x.ctx, name=f'sgd.{i}.v')
            self.params.append((x, v_i))

    def step(self):
        for x0, v0 in self.params:
            if x0.grad is None:
                continue
            v1, x1 = sgd(x0, x0.grad, v0, self._lr, self._momentum)
            x0.update(x1)
            v0.update(v1)
