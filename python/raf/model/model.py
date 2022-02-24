# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Interactive interface for training & inference."""
from collections import OrderedDict

from raf._core import cacher
from raf._core.core_utils import bfs, get_attr, get_named_attr, set_module
from raf._core.ndarray import ndarray
from raf.model.trace import _get_trace_record


@set_module("raf")
class BaseModel:
    def __init__(self):
        super(BaseModel, self).__init__()
        self.__is_train = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def train_mode(self, recursive=True):
        _set_is_train(self, value=True, recursive=recursive)

    def infer_mode(self, recursive=True):
        _set_is_train(self, value=False, recursive=recursive)

    def _internal(self, *args, **kwargs):
        raise NotImplementedError

    def _state(self):
        raise NotImplementedError

    def state(self, prefix="", recursive=True):
        return _get_param_dict(self, prefix=prefix, recursive=recursive)

    def to(self, *, device=None, dtype=None):  # pylint: disable=invalid-name
        raise NotImplementedError


class Model(BaseModel, cacher.Cacher):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        build, self.__fwd_train, self.__fwd_infer = _extract_methods(self)
        build(*args, **kwargs)
        cacher.enable(self)  # Cache is set up after the model is built

    def __call__(self, *args, **kwargs):
        forward = (
            self.__fwd_train
            if self._BaseModel__is_train  # pylint: disable=no-member
            else self.__fwd_infer
        )
        return forward(*args, **kwargs)

    def train_mode(self, recursive=True):
        super(Model, self).train_mode(recursive=recursive)
        cacher.invalidate(self, include_self=True, recursive=True)

    def infer_mode(self, recursive=True):
        super(Model, self).infer_mode(recursive=recursive)
        cacher.invalidate(self, include_self=True, recursive=True)

    def _internal(self, *args, **kwargs):
        """
        Get internal IR information.
        TODO(yzhliu): we may consider APIs like following in the future.
        model.set_input(...)
        internal = model._internal()

        Parameters
        ----------
        args : raf.ndarray
            The input data of the model.
        kwargs : named inputs
            Currently not supported.

        Returns
        -------
        record: _TraceRecord
            The traced record.
            Get raf module by record.mod, parameters by record.named_params.
        """
        fwd_func = (
            self.__fwd_train
            if self._BaseModel__is_train  # pylint: disable=no-member
            else self.__fwd_infer
        )
        pyfunc = fwd_func.__wrapped__
        # TODO(hgt312): varargs and kwargs
        args = [self] + list(args)

        record = _get_trace_record(pyfunc, args, kwargs)
        m_mod = record.mod
        r_func = m_mod["main"]
        # already cached
        if len(record.requires_grads) != 0:
            assert len(r_func.params) == len(record.requires_grads)
            return record
        # do not care `requires_grads`
        if len(args) + len(kwargs) == 1:
            return record

        assert len(r_func.params) + 1 == len(args) + len(kwargs) + len(record.named_params)
        arg_index = 1
        for var_node in r_func.params:
            var_name = var_node.name_hint
            if var_name in record.named_params:
                record.requires_grads.append(record.named_params[var_name].requires_grad)
            elif var_name in kwargs:
                arg = kwargs[var_name]
                if isinstance(arg, ndarray):
                    record.requires_grads.append(arg.requires_grad)
                else:
                    record.requires_grads.clear()
                    return record
            else:
                arg = args[arg_index]
                if isinstance(arg, ndarray):
                    record.requires_grads.append(arg.requires_grad)
                    arg_index += 1
                else:
                    record.requires_grads.clear()
                    return record
        return record

    def _state(self):
        return _get_attr_params_key_value(self)

    def to(self, *, device=None, dtype=None, invalidate=True):  # pylint: disable=arguments-differ
        for model in _get_model_dict(self, prefix="", recursive=True).values():
            for name, param in _get_attr_params_key_value(model).items():
                param = param.to(device=device, dtype=dtype)
                setattr(model, name, param)
        if invalidate:
            cacher.invalidate(self, include_self=True, recursive=True)


# pylint: disable=protected-access


def _get_attr_models_key_value(model):
    return get_named_attr(model, check=lambda x: isinstance(x, BaseModel))


def _get_attr_models_value(model):
    return get_attr(model, check=lambda x: isinstance(x, BaseModel))


def _get_attr_params_key_value(model):
    return get_named_attr(model, check=lambda x: isinstance(x, ndarray))


def _get_attr_params_value(model):
    return get_attr(model, check=lambda x: isinstance(x, ndarray))


def _set_is_train(root_model, *, value, recursive):
    if not hasattr(root_model, "_BaseModel__is_train"):
        return

    def on_pop(model):
        object.__setattr__(model, "_BaseModel__is_train", value)
        for param in _get_attr_params_value(model):
            # TODO(@junrushao1994): maybe invalidate param's other parents?
            param.requires_grad = value

    bfs([root_model], on_pop, on_next=_get_attr_models_value, recursive=recursive)


def _get_param_dict(root_model, *, prefix, recursive):
    model_prefix = {root_model: prefix}
    result = OrderedDict()

    def on_pop(model):
        prefix = model_prefix[model]
        if prefix != "":
            prefix = prefix + "."
        for name, item in model._state().items():
            result[prefix + name] = item
        for name, item in _get_attr_models_key_value(model).items():
            model_prefix[item] = prefix + name

    bfs([root_model], on_pop, on_next=_get_attr_models_value, recursive=recursive)
    return result


def _get_model_dict(root_model, *, prefix, recursive):
    model_prefix = {root_model: prefix}
    result = OrderedDict()

    def on_pop(model):
        prefix = model_prefix[model]
        result[prefix] = model
        if prefix != "":
            prefix = prefix + "."
        for name, item in _get_attr_models_key_value(model).items():
            model_prefix[item] = prefix + name

    bfs([root_model], on_pop, on_next=_get_attr_models_value, recursive=recursive)
    return result


def _extract_methods(model):
    build = get_attr(model, name="build", check=callable)
    fwd_train = get_attr(model, name="forward", check=callable)
    fwd_infer = get_attr(model, name="forward_infer", check=callable)
    if not build:
        raise KeyError("Please implement build() method")
    if not fwd_train:
        raise KeyError("Please implement forward() method")
    if not fwd_infer:
        fwd_infer = fwd_train
    return build, fwd_train, fwd_infer


# pylint: enable=protected-access
