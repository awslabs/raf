# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGD optimizer."""
# pylint: disable=too-many-statements,too-many-instance-attributes
import numpy as np

from raf._core.core_utils import get_chained_attr
from raf._core.ndarray import ndarray, array
from raf.model import trace, Model, trace_mutate_attr
from raf.model.trace import _get_func_inputs
from raf._op import imp
from raf._op.sym import multiply, add, subtract, strided_slice, cast
from .. import distributed as dist
from .data_parallel import with_data_parallel
from ..distributed.op import allgather
from .optim import with_autodiff
from .utils import has_grad, split_ndarray_with_padding


# pylint: disable=too-few-public-methods
class SGD:
    """Optimizer : stochastic gradient descent

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
            assert isinstance(x, ndarray), "Only `raf.ndarray' can be optimized!"
            npa = np.zeros(x.shape, dtype=x.dtype)
            v_i = ndarray(npa, device=x.device, name=f"sgd.{i}.v")
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
    """Optimizer : stochastic gradient descent

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

    def get_model_dtype(model):
        """A helper function to determine the parameter dtype by referring to
        the first floating type parameter.

        Parameters
        ----------
        model: Model
            The model to be evaluated.
        """
        for param in model.state().values():
            if "float" in param.dtype:
                return param.dtype
        return "float32"

    def decorator(model):
        class SGDWrapper(Model):
            """sgd wrapper model

            Parameters
            ----------
            model: Model
                The forward model
            """

            # pylint: disable=attribute-defined-outside-init, protected-access, too-many-locals
            # pylint: disable=missing-function-docstring
            def build(self, model):
                self.model = model
                self.ad_model = with_data_parallel(with_autodiff(model))
                self.learning_rate = array(learning_rate, dtype="float32")
                self.momentum = array(momentum, dtype="float32")

                # Determine the parameter dtype by referring to the first floating type parameter.
                self.dtype = get_model_dtype(self.model)

                # Whether we have SGD weight buffers that are differernt from model parameters.
                # In the case of training a float32 model on single device, the SGD weights
                # are always identical to the model parameters, so we don't need to create
                # additional buffers in SGD status.
                self.has_sgd_w = False

                dcfg = dist.get_config()
                comm = dist.get_communicator()
                self.params = {}
                for name, param in self.model.state().items():
                    # For each tensor "param" that requires gradient (i.e., training weights),
                    # create a tensor "param.sgd_v" to be its SGD variant.
                    # TODO(@anijain2305): We might mark requires_grad to non-float parameters
                    # which is incorrect. We should improve the gradient pass for better
                    # has_grad results.
                    if param.requires_grad and "float" in param.dtype:
                        assert isinstance(param, ndarray), "Only `raf.ndarray` can be optimized!"

                        # By default we directly use the model parameter as the SGD weight.
                        v_w = param

                        status_shape = param.shape
                        if dcfg.zero_opt_level:
                            # If optimizer status partitioning is enable, then the first axis of
                            # variant and weight is partitioned to 1/n. Accordingly, we have to
                            # also keep a param.w (size 1/n) locally.

                            # Pad and copy a slice of weight to be the SGD statues.
                            param_nd = param.to(device="cpu")
                            if "float" in param.dtype and param.dtype != "float32":
                                param_nd = param_nd.to(dtype="float32")
                            slice_param = split_ndarray_with_padding(param_nd, comm.size)[comm.rank]
                            v_w = ndarray(
                                slice_param,
                                device=param.device,
                                name=f"{name}.sgd_w",
                                dtype="float32",
                            )
                            self.has_sgd_w = True

                            # The SGD status shape of this parameter is now 1/n.
                            status_shape = slice_param.shape
                        elif "float" in param.dtype and param.dtype != "float32":
                            # Maintain float32 weights for accuracy.
                            v_w = ndarray(
                                param.to(dtype="float32"),
                                device=param.device,
                                name=f"{name}.sgd_w",
                                dtype="float32",
                            )
                            self.has_sgd_w = True

                        # Maintain a weight copy if it is differernt as the model parameter.
                        if self.has_sgd_w:
                            setattr(self, f"{name}.sgd_w", v_w)

                        # Initialize variants according to the status shape.
                        v_i = ndarray(
                            np.zeros(status_shape, dtype="float32"),
                            device=param.device,
                            name=f"{name}.sgd_v",
                        )
                        setattr(self, f"{name}.sgd_v", v_i)
                        self.params[param._ndarray__handle] = (name, param, v_w, v_i)

                if self.has_sgd_w:
                    # TODO(issue 758): Remove this and in-place update parameters.
                    self.zero = array(0, dtype=self.dtype)

            @trace
            def forward(self, dy, *args, **kwargs):
                y, dxs = self.ad_model(dy, *args, **kwargs)
                record = self.ad_model._internal(dy, *args, **kwargs)
                inputs = _get_func_inputs(record, [dy, *args], kwargs)
                inputs = inputs[1:]  # remove dy
                dcfg = dist.get_config()
                comm = dist.get_communicator()
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, weight, sgd_w, sgd_v = self.params[param]
                        assert "float" in sgd_w.dtype, "Non-float parameter is not learnable"

                        # Cast gradient to float32 if necessary.
                        if self.dtype != "float32":
                            dxi = cast(dxi, "float32")

                        # Inplace update the local SGD variant and weight (float32).
                        new_sgd_v = add(multiply(self.momentum, sgd_v), dxi, out=sgd_v)
                        new_sgd_w = subtract(
                            sgd_w, multiply(self.learning_rate, new_sgd_v), out=sgd_w
                        )

                        # Cast the updated SGD weight to the model parameter dtype.
                        if self.dtype != "float32":
                            new_sgd_w = cast(new_sgd_w, self.dtype)

                        # If the SGD status is partitioned, use all-gather to sync
                        # the updated weights.
                        if dcfg.zero_opt_level > 0:
                            new_sgd_w = allgather(new_sgd_w, axis=0)
                            # Slice to remove the zero-padding if needed.
                            if sgd_w.shape[0] * comm.size > weight.shape[0]:
                                new_sgd_w = strided_slice(new_sgd_w, [0], [weight.shape[0]], [1])

                        # Update the model parameter.
                        new_weight = (
                            add(new_sgd_w, self.zero, out=weight) if self.has_sgd_w else new_sgd_w
                        )

                        # Put the updated weight to the model output to avoid being dead code.
                        param_model = get_chained_attr(self.model, name.split(".")[:-1])
                        trace_mutate_attr(param_model, name.split(".")[-1], new_weight)
                return y

        return SGDWrapper(model)

    return decorator
