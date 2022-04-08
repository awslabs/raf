# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for getting models from PyTorch model zoos."""
# pylint: disable=import-outside-toplevel, protected-access, too-many-locals
import os

import numpy as np
import torch

from tvm import relay

from .._core.module import IRModule
from .._core.ndarray import array, Symbol
from .._op import sym
from .._ffi.pass_ import InferType, ExprAppend, ExtractBinding
from ..frontend.model import FrameworkModel
from ..frontend.pytorch import from_pytorch
from .. import optim


class ConvertNLPContext:
    """The context to deal with TOKENIZERS_PARALLELISM."""

    def __init__(self):
        self.tokenizers_parallelism = None

    def __enter__(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            self.tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __exit__(self, ptype, value, trace):
        if self.tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]


def get_raf_func_output_var(func):
    """A helper function to get the output Var of the given function."""
    body = func.body
    while not isinstance(body, relay.Var):
        if isinstance(body, relay.Let):
            body = body.body
        else:
            raise NotImplementedError("Not supported type: ", type(body))
    return body


def get_transformer_model(name, batch_size=8, seq_length=128, dtype="float32"):
    """Load a Huggingface transformer model and convert to RAF.

    Parameters
    ----------
    name: str
        Name of the model.

    batch_size: int
        Batch size. Default 8.

    seq_length: int
        Sequence length. Default 128.

    dtype: str
        Data type. Default "float32".

    Returns
    -------
    model_n_shape: Tuple[raf.model, Tuple[int, ...]]
        The converted RAF model and forward output shape.
    """
    import transformers

    config = transformers.AutoConfig.from_pretrained(name)
    return get_transformer_model_by_config(config, batch_size, seq_length, dtype)


def get_transformer_model_by_config(config, batch_size=8, seq_length=128, dtype="float32"):
    """Load a Huggingface transformer model and convert to RAF.

    Parameters
    ----------
    config: transformers.configuration_utils.PretrainedConfig
        Model configuration.

    batch_size: int
        Batch size. Default 8.

    seq_length: int
        Sequence length. Default 128.

    dtype: str
        Data type. Default "float32".

    Returns
    -------
    model_n_shape: Tuple[raf.model, Tuple[int, ...]]
        The converted RAF model and forward output shape.
    """
    import transformers

    # Load the PyTorch model.
    with ConvertNLPContext():
        assert hasattr(config, "architectures"), '"architectures" is missing in the config'
        model_cls = config.architectures[0]
        config.use_cache = False  # Disable model cache to avoid unnecessary model outputs.
        assert hasattr(transformers, model_cls), "%s is not supported in transformers" % model_cls
        t_model = getattr(transformers, model_cls)(config)

        input_shape = [batch_size, seq_length]

        np_x = np.random.randint(0, 10000, input_shape)
        t_x = torch.tensor(np_x)
        t_model.eval()
        if dtype == "float16":
            t_model.half()
        t_y = t_model(t_x)[0]

        torch.cuda.empty_cache()

    # Convert to RAF model.
    try:
        r_x = array(t_x)
        r_model = from_pytorch(t_model, {"input_ids": (input_shape, "int64")})
        record = r_model._internal(r_x)
        mod = record.mod
        mod = InferType()(mod)
        func = mod["main"]
        ret_var = get_raf_func_output_var(func)
        if isinstance(ret_var.checked_type, relay.TupleType):
            ret = Symbol.from_expr(ret_var)
            ret = ret[0]
            ret = ExtractBinding(ret._Symbol__handle, [ret_var])
            new_body = ExprAppend(func.body, ret)
            new_func = relay.Function(func.params, new_body)
            new_mod = IRModule.from_expr(new_func)
            r_model = FrameworkModel(
                new_mod,
                new_mod,
                r_model._FrameworkModel__arg_params,
                r_model._FrameworkModel__aux_params,
            )
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to convert model to RAF: %s" % (str(err)))

    return r_model, t_y.shape


def get_torchvision_model(name, batch_size=8, image_size=(224, 224), dtype="float32"):
    """Load a torchvision model and convert to RAF.

    Parameters
    ----------
    name: str
        Name of the model.

    batch_size: int
        Batch size. Default 8.

    image_size: Tuple[int, int]
        Image size. Default (224, 224)

    dtype: str
        Data type. Default "float32".

    Returns
    -------
    model_n_shape: Tuple[raf.model, Tuple[int, ...]]
        The converted RAF model and forward output shape.
    """
    import torchvision

    input_shape = (batch_size, 3, *image_size)
    t_model = getattr(torchvision.models, name)(pretrained=True)
    t_model.eval()
    if dtype == "float16":
        t_model.half()

    t_x = torch.tensor(np.random.randn(*input_shape).astype(dtype))
    t_y = t_model(t_x)

    torch.cuda.empty_cache()

    try:
        m_model = from_pytorch(t_model, {"input0": ((input_shape, dtype))})
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to convert model to RAF: %s" % (str(err)))

    return m_model, t_y.shape


def append_loss_n_optimizer(model, args, out_shape, y_true, optimizer="sgd"):
    """Append log_softmax + nll_loss, followed by an optimizer.

    Parameters
    ----------
    model: raf.model
        The model to append loss and optimizer.

    args: List[raf.array]
        The input arguments.

    out_shape: Tuple[int, ...]
        The forward output shape.

    y_true: raf.array
        The ground truth.

    optimizer: str
        The optimizer name. Default "sgd".

    Returns
    -------
    optimizer: raf.Model
        The model with loss and optimizer.
    """
    out = model.record(*args)

    # Reshape output if necessary.
    if len(out_shape) != len(y_true.shape):
        new_shape = [np.prod(out_shape[:-1]).tolist(), out_shape[-1]]
        out = sym.reshape(out, new_shape)
    y_pred = sym.log_softmax(out)
    loss = sym.nll_loss(y_true, y_pred)

    model_w_loss = model + loss
    model_w_loss.train_mode()

    if optimizer == "sgd":
        return optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(model_w_loss)
    if optimizer == "lans":
        return optim.lans.with_lans()(model_w_loss)
    raise ValueError("Unrecognized optimizer: %s" % optimizer)
