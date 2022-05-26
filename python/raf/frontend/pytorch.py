# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend that converts PyTorch models to RAF models via Relay."""
# pylint: disable=too-many-locals
from collections import OrderedDict
import os
import hashlib
import torch

from raf import distributed as dist
from .._core.ndarray import ndarray
from .._lib import relay
from .._ffi.pass_ import FromRelay, SwitchTrainOp, validate_relay_param_name
from ..frontend.model import FrameworkModel


def trace_model(model, shape_dict):
    """Trace PyTorch model.

    Parameters
    ----------
    model: torch.nn.Module
        The PyTorch module to be converted.

    shape_dict: Dict[str,
                     Union[Tuple[Tuple[int, ...], str],
                           Tuple[Tuple[int, ...], str, int]]
        A map from input name to its shape, type, and maximal value (optional).

    Returns
    -------
    model: ScriptedModel
        PyTorch scripted model.
    """

    class TraceWrapper(torch.nn.Module):
        """A wrapper to process the forward output. This is required for object detection
        models which have multiple outputs.
        """

        # pylint: disable=missing-function-docstring, abstract-method ,arguments-differ

        # Enforce the output order of object detection models.
        od_model_output_keys = ["boxes", "scores", "labels", "masks"]

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *inputs):
            out = self.model(*inputs)
            if isinstance(out, list):
                ordered_outs = [out[0][key] for key in self.od_model_output_keys if key in out[0]]
                return tuple(ordered_outs)
            if isinstance(out, dict):
                return out.to_tuple()
            return out

        @property
        def dtype(self):
            for param in model.parameters():
                if param.dtype.is_floating_point:
                    return param.dtype
            return torch.float32

    def inner(model, shape_dict):
        """Wrap the tracing process so that we could empty PyTorch CUDA cache afterward."""
        model = TraceWrapper(model)
        model.eval()

        # By default we trace the model on CPU.
        device = "cpu"

        # Some float16 ops are only available on GPU.
        if model.dtype != torch.float32:
            if not torch.cuda.is_available():
                raise RuntimeError("Trace PyTorch model with dtype %s requires GPU" % model.dtype)
            comm = dist.get_communicator()
            device = "cuda:" + str(comm.local_rank)

        example_inputs = []
        for _, input_info in shape_dict.items():
            input_shape = input_info[0]
            input_type = input_info[1]
            if input_type.startswith("int64"):
                max_val = 10000 if len(input_info) == 2 else input_info[2]
                input_data = torch.randint(max_val + 1, input_shape, device=device)
            elif input_type.startswith("float"):
                input_data = torch.randn(
                    input_shape, dtype=getattr(torch, input_type), device=device
                )
            else:
                raise ValueError("Unsupported input type %s" % input_type)
            example_inputs.append(input_data)

        with torch.no_grad():
            model.to(device=device)
            scripted_model = torch.jit.trace(model, tuple(example_inputs)).eval()

        if device.startswith("cuda"):
            model.to(device="cpu")
            scripted_model = scripted_model.to(device="cpu")

        return scripted_model

    scripted_model = inner(model, shape_dict)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return scripted_model


def from_pytorch(model, shape_dict, model_file=None, hash_file=None):
    """Load PyTorch model and convert into RAF via Relay.

    Parameters
    ----------
    model: torch.nn.Module
        The PyTorch module to be converted.

    shape_dict: Dict[str,
                     Union[Tuple[Tuple[int, ...], str],
                           Tuple[Tuple[int, ...], str, int]]
        A map from input name to its shape, type, and maximal value (optional).

    model_file: str
        The file that stores the scripted model

    hash_file: str
        The file that stores the scripted model hash
    Returns
    -------
    model: FrameworkModel
        The converted FrameworkModel.
    """
    if model_file is not None and hash_file is not None:
        model_hash = hashlib.md5(str(model).encode(encoding="UTF-8")).hexdigest()
        if os.path.exists(model_file) and os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as hashf:
                    mhash = hashf.read()
                    if mhash != model_hash:
                        raise RuntimeError("Hash check failed")
                    scripted_model = torch.jit.load(model_file)
            except:
                raise RuntimeError("Loading scripted model failed")
        else:
            scripted_model = trace_model(model, shape_dict)
            scripted_model.eval()
            scripted_model.save(model_file)
            with open(hash_file, "w") as hashf:
                hashf.write(model_hash)
    else:
        scripted_model = trace_model(model, shape_dict)
    shape_list = []
    for input_name, input_info in list(shape_dict.items()):
        input_shape = input_info[0]
        input_type = input_info[1]
        shape_list.append((input_name, (input_shape, input_type)))
    relay_mod, relay_params = relay.frontend.from_pytorch(scripted_model, shape_list)
    meta_mod = FromRelay()(relay_mod)
    meta_params = OrderedDict()
    for var in relay_mod["main"].params:
        name = var.name_hint
        if name in relay_params:
            meta_params[validate_relay_param_name(name)] = ndarray(relay_params[name].numpy())
    # relay_params may contain unused parameters, which are not present in meta_params
    assert len(meta_params) <= len(relay_params)
    return FrameworkModel(SwitchTrainOp(True)(meta_mod), meta_mod, meta_params, {})
