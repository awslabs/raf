# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name, missing-function-docstring, too-many-instance-attributes, too-many-locals, too-many-statements, protected-access, too-many-arguments, too-many-branches
"""LANS optimizer."""
import numpy as np

from raf._core.core_utils import get_chained_attr
from raf._core.ndarray import array, ndarray
from raf.model import trace, Model, trace_mutate_attr
from raf.model.trace import _get_func_inputs
from raf._op import sym as _op
from raf._op import imp
from .. import distributed as dist
from .data_parallel import with_data_parallel
from ..distributed.op import allgather
from .optim import with_autodiff
from .utils import has_grad, split_ndarray_with_padding


# pylint: disable=too-few-public-methods
class LANS:
    """Optimizer : LANS
    # References
    - Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes.
      http://arxiv.org/abs/2006.13484

    Parameters
    ----------
    lr: Optional[Float]
        Learning rate. Default: 1e-3

    betas: Optional[Tuple[Float, Float]]
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)

    eps: Optional[Float]
        Term added to the denominator to improve numerical stability. Default: 1e-6

    weight_decay: Optional[Float]
        Weight decay (L2 penalty). Default: 0.01
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        bias_correction=True,
        grad_averaging=True,
        mode=True,
        normalize_grad=True,
    ):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.grad_averaging = grad_averaging
        self.mode = mode
        self.normalize_grad = normalize_grad
        self.params = []
        self._step = 1
        for i, x in enumerate(params):
            assert isinstance(x, ndarray), "Only `raf.ndarray' can be optimized!"
            npa = np.zeros(x.shape, dtype=x.dtype)
            m_i = ndarray(npa, device=x.device, name=f"lans.{i}.m")
            v_i = ndarray(npa, device=x.device, name=f"lans.{i}.v")
            self.params.append((x, m_i, v_i))

    def step(self):
        """Update the parameters with gradients."""
        tensor_list = []
        g_list = []
        x_list = []
        m_list = []
        v_list = []
        for x, m, v in self.params:
            if x.grad is None:
                continue
            g_list.append(x.grad)
            x_list.append(x)
            m_list.append(m)
            v_list.append(v)
        step = array(self._step, dtype="float32", device="cuda", name="step")
        tensor_list = g_list + x_list + m_list + v_list
        imp.lans(
            tensor_list,
            step,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.bias_correction,
            self.weight_decay,
            self.grad_averaging,
            self.mode,
            self.normalize_grad,
        )
        self._step += 1


def with_lans(
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01,
    bias_correction=True,
    grad_averaging=True,
    mode=True,
    normalize_grad=True,
):
    """Optimizer : LANS
    # References
    - Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes.
      http://arxiv.org/abs/2006.13484

    Parameters
    ----------
    lr: Optional[Float]
        Learning rate. Default: 1e-3

    betas: Optional[Tuple[Float, Float]]
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)

    eps: Optional[Float]
        Term added to the denominator to improve numerical stability. Default: 1e-6

    weight_decay: Optional[Float]
        Weight decay (L2 penalty). Default: 0.01

    Returns
    ret : function
        The wrapper which wraps a model with LANS
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
        class LANSWrapper(Model):
            """LANS wrapper model

            Parameters
            ----------
            model: the forward model
            """

            # pylint: disable=attribute-defined-outside-init
            def build(self, model):
                self.model = model
                self.ad_model = with_data_parallel(with_autodiff(model))
                self.bias_correction = bias_correction
                self.mode = mode
                self.normalize_grad = normalize_grad
                self.lr = lr
                self.eps = eps
                self.weight_decay = weight_decay
                self.grad_averaging = grad_averaging
                self.beta1 = betas[0]
                self.beta2 = betas[1]
                # Determine the parameter dtype by referring to the first floating type parameter.
                self.dtype = get_model_dtype(self.model)
                self.zero = array(0.0, dtype=self.dtype)
                self.one = array(1.0, dtype="float32")
                # mutable params: global step, and running averages
                device = None
                dcfg = dist.get_config()
                comm = dist.get_communicator()
                self.params = {}
                for name, param in self.model.state().items():
                    if param.requires_grad is True:
                        if device is None:
                            device = param.device
                        else:
                            assert device == param.device
                        assert isinstance(param, ndarray), "Only `raf.ndarray` can be optimized!"
                        # If optimizer status partitioning is enable, then the first axis of
                        # variant and weight is partitioned to 1/n. Accordingly, we have to
                        # also keep a param.w (size 1/n) locally.
                        part_shape = param.shape
                        if dcfg.zero_opt_level:
                            # Pad and copy a slice of weight.
                            param_nd = param.to(device="cpu")
                            if "float" in param.dtype and param.dtype != "float32":
                                param_nd = param_nd.to(dtype="float32")
                            slice_param = split_ndarray_with_padding(param_nd, comm.size)[comm.rank]
                            param_part = ndarray(
                                slice_param,
                                device=param.device,
                                name=f"{name}.lans_w",
                                dtype="float32",
                            )
                            weight = param_part
                            setattr(self, f"{name}.lans_w", weight)
                            part_shape = slice_param.shape
                        elif "float" in param.dtype and param.dtype != "float32":
                            weight = ndarray(
                                param.to(dtype="float32"),
                                device=param.device,
                                name=f"{name}.lans_w",
                                dtype="float32",
                            )
                            setattr(self, f"{name}.lans_w", weight)
                        else:
                            weight = param
                        npa = np.zeros(part_shape, dtype="float32")
                        m_i = array(npa, device=device, name=f"{name}.m")
                        v_i = array(npa, device=device, name=f"{name}.v")
                        setattr(self, f"{name}.m", m_i)
                        setattr(self, f"{name}.v", v_i)
                        self.params[param._ndarray__handle] = (name, param, weight, m_i, v_i)
                assert device is not None
                self.step = array(0.0, dtype="float32", device=device, name="step")

            @trace
            def forward(self, dy, *args, **kwargs):
                dcfg = dist.get_config()
                comm = dist.get_communicator()
                y, dxs = self.ad_model(dy, *args, **kwargs)
                record = self.ad_model._internal(dy, *args, **kwargs)
                inputs = _get_func_inputs(record, [dy, *args], kwargs)
                inputs = inputs[1:]  # remove dy
                # update step
                next_step = _op.add(self.step, self.one, out=self.step)
                trace_mutate_attr(self, "step", next_step)
                # apply LANS for each param
                tensor_list = []
                g_list = []
                x_list = []
                m_list = []
                v_list = []
                ntensor = 0
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, p, w, m, v = self.params[param]
                        if "float" not in w.dtype:
                            continue

                        g_list.append(dxi)
                        x_list.append(w)
                        m_list.append(m)
                        v_list.append(v)
                        ntensor += 1

                if self.dtype != "float32":
                    fp32_g = _op.group_cast(g_list, "float32")
                    g_list = []
                    for i in range(ntensor):
                        g_list.append(fp32_g[i])

                tensor_list = g_list + x_list + m_list + v_list
                output_list = _op.lans(
                    tensor_list,
                    next_step,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.bias_correction,
                    self.weight_decay,
                    self.grad_averaging,
                    self.mode,
                    self.normalize_grad,
                )

                out_idx = 0
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, p, w, m, v = self.params[param]
                        if "float" in w.dtype:
                            new_w = output_list[out_idx + ntensor]
                            if self.dtype != "float32":
                                new_w = _op.cast(new_w, self.dtype)

                            next_m = output_list[out_idx + 2 * ntensor]
                            next_v = output_list[out_idx + 3 * ntensor]
                            param_model = get_chained_attr(self.model, name.split(".")[:-1])
                            if dcfg.zero_opt_level > 0:
                                new_weight = allgather(new_w, axis=0)
                                # Slice to remove the zero-padding if needed.
                                if w.shape[0] * comm.size > p.shape[0]:
                                    new_weight = _op.strided_slice(
                                        new_weight, [0], [p.shape[0]], [1]
                                    )
                                next_w = _op.add(new_weight, self.zero, out=p)
                            else:
                                if self.dtype != "float32":
                                    next_w = _op.add(new_w, self.zero, out=p)
                                else:
                                    # LANS inplace upates the weight
                                    # So the new  weight is just the input weight
                                    next_w = new_w

                            trace_mutate_attr(param_model, name.split(".")[-1], next_w)
                            trace_mutate_attr(self, f"{name}.m", next_m)
                            trace_mutate_attr(self, f"{name}.v", next_v)
                            out_idx += 1
                return y

        return LANSWrapper(model)

    return decorator
