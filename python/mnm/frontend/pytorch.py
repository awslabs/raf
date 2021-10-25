"""The frontend that converts PyTorch models to Meta models via Relay."""
# pylint: disable=too-many-locals
from collections import OrderedDict
import torch

from .._core.ndarray import ndarray
from .._lib import relay
from .._ffi.pass_ import FromRelay, validate_relay_param_name
from ..frontend.model import FrameworkModel


def from_pytorch(model, shape_dict):
    """Load PyTorch model and convert into Meta via Relay.

    Parameters
    ----------
    model: torch.nn.Module
        The PyTorch module to be converted.

    shape_dict: Dict[str, Tuple[Tuple[int, ...], str]]
        A map from input name to its shape and type. Note that we currently only support
        the model with a single input.

    Returns
    -------
    model: FrameworkModel
        The converted FrameworkModel.
    """
    if len(shape_dict) > 1:
        raise RuntimeError(
            "Do not support PyTorch model with multiple inputs (%d) yet" % len(shape_dict))
    input_name, (input_shape, input_type) = list(shape_dict.items())[0]

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

        def forward(self, inp):
            out = self.model(inp)
            if isinstance(out, list):
                ordered_outs = [out[0][key] for key in self.od_model_output_keys if key in out[0]]
                return tuple(ordered_outs)
            if isinstance(out, dict):
                return out.to_tuple()
            return out

    model = TraceWrapper(model)
    model.eval()
    device = "cpu"

    if input_type.startswith("float"):
        if input_type.startswith("float16"):
            device = "cuda" # Some float16 ops are only available on GPU.
        input_data = torch.randn(input_shape, dtype=getattr(torch, input_type), device=device)
    else:
        assert input_type.startswith("int64"), "Unsupported input type %s" % input_type
        input_data = torch.randint(10000, input_shape)

    with torch.no_grad():
        model.to(device=device)
        model(input_data)
        scripted_model = torch.jit.trace(model, input_data).eval()
        scripted_model.eval()

    shape_list = [(input_name, input_shape)]
    relay_mod, relay_params = relay.frontend.from_pytorch(scripted_model, shape_list)
    meta_mod = FromRelay()(relay_mod)
    meta_params = OrderedDict()
    for var in relay_mod["main"].params:
        name = var.name_hint
        if name in relay_params:
            meta_params[validate_relay_param_name(name)] = ndarray(relay_params[name].numpy())
    # relay_params may contain unused parameters, which are not present in meta_params
    assert len(meta_params) <= len(relay_params)
    return FrameworkModel(meta_mod, meta_mod, meta_params, {})
