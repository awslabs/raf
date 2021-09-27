"""SGD optimizer."""
import numpy as np

from mnm._core.core_utils import get_chained_attr
from mnm._core.ndarray import ndarray, array
from mnm.model import trace, Model, trace_mutate_attr
from mnm.model.trace import _get_func_inputs
from mnm._op import imp
from mnm._op.sym import multiply, add, subtract, strided_slice
from .. import distributed as dist
from .data_parallel import with_data_parallel
from ..distributed.op import allgather
from .optim import with_autodiff
from .utils import has_grad, split_ndarray_with_padding


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
            # pylint: disable=missing-function-docstring
            def build(self, model):
                self.model = model
                self.ad_model = with_data_parallel(with_autodiff(model))
                self.learning_rate = array(learning_rate, dtype='float32')
                self.momentum = array(momentum, dtype='float32')

                # TODO(issue 758): Remove this and in-place update parameters.
                self.zero = array(0, dtype="float32")

                dctx = dist.get_context()
                self.params = {}
                for name, param in self.model.state().items():
                    # For each tensor "param" that requires gradient (i.e., training weights),
                    # create a tensor "param.sgd_v" to be its SGD variant.
                    if param.requires_grad is True:
                        assert isinstance(param, ndarray), "Only `mnm.ndarray` can be optimized!"

                        # If optimizer status partitioning is enable, then the first axis of
                        # variant and weight is partitioned to 1/n. Accordingly, we have to
                        # also keep a param.w (size 1/n) locally.
                        part_shape = param.shape
                        if dctx.zero_opt_level:
                            # Pad and copy a slice of weight to be the SGD statues.
                            param_np = param.to(device="cpu")
                            slice_param = split_ndarray_with_padding(param_np, dctx.size)[dctx.rank]
                            v_w = ndarray(slice_param, device=param.device, name=f'{name}.sgd_w')
                            setattr(self, f'{name}.sgd_w', v_w)
                            part_shape = slice_param.shape
                        else:
                            v_w = param

                        v_i = ndarray(np.zeros(part_shape, dtype=param.dtype), device=param.device,
                                      name=f'{name}.sgd_v')
                        setattr(self, f'{name}.sgd_v', v_i)
                        self.params[param._ndarray__handle] = (name, param, v_w, v_i)

            @trace
            def forward(self, dy, *args, **kwargs):
                y, dxs = self.ad_model(dy, *args, **kwargs)
                record = self.ad_model._internal(dy, *args, **kwargs)
                inputs = _get_func_inputs(record, [dy, *args], kwargs)
                inputs = inputs[1:]  # remove dy
                dctx = dist.get_context()
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, weight, sgd_w, sgd_v = self.params[param]
                        # TODO(@anijain2305): Improve the gradient pass for better has_grad results.
                        if "float" not in sgd_w.dtype:
                            continue

                        # Inplace update SGD variant and weight.
                        new_sgd_v = add(multiply(self.momentum, sgd_v), dxi, out=sgd_v)
                        new_sgd_w = subtract(sgd_w, multiply(self.learning_rate, new_sgd_v),
                                             out=sgd_w)

                        # If the SGD status is partitioned, use all-gather to sync
                        # the updated weights.
                        if dctx.zero_opt_level > 0:
                            new_weight = allgather(new_sgd_w, axis=0)
                            # Slice to remove the zero-padding if needed.
                            if sgd_w.shape[0] * dctx.size > weight.shape[0]:
                                new_weight = strided_slice(new_weight, [0], [weight.shape[0]], [1])
                            new_weight = add(new_weight, self.zero, out=weight)
                        else:
                            new_weight = new_sgd_w

                        # Put the updated weight to the model output to avoid being dead code.
                        param_model = get_chained_attr(self.model, name.split('.')[:-1])
                        trace_mutate_attr(param_model, name.split('.')[-1], new_weight)
                return y
        return SGDWrapper(model)
    return decorator
