# pylint: disable=missing-class-docstring,missing-function-docstring
"""Interactive interface for training & inference."""
import inspect
from collections import OrderedDict

from mnm._core import cacher
from mnm._core.core_utils import bfs, get_attr, get_named_attr, set_module
from mnm._core.ndarray import ndarray

from mnm.model.trace import _get_trace_record


@set_module("mnm")
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

    def state(self, prefix="", recursive=True):
        return _get_param_dict(self, prefix=prefix, recursive=recursive)

    def to(self, *, device=None, dtype=None):  # pylint: disable=invalid-name
        # TODO(@junrushao1994): do we control cache invalidation?
        for model in _get_model_dict(self, prefix="", recursive=True).values():
            for name, param in _get_attr_params_key_value(model).items():
                param = param.to(device=device, dtype=dtype)
                setattr(model, name, param)


class Model(BaseModel, cacher.Cacher):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        build, self.__fwd_train, self.__fwd_infer = _extract_methods(self)
        build(*args, **kwargs)
        cacher.enable(self)  # Cache is set up after the model is built

    def __call__(self, *args, **kwargs):
        forward = self.__fwd_train if self._BaseModel__is_train else self.__fwd_infer  # pylint: disable=no-member
        return forward(*args, **kwargs)

    def train_mode(self, recursive=True):
        super(Model, self).train_mode(recursive=recursive)
        cacher.invalidate(self, include_self=False, recursive=True)

    def infer_mode(self, recursive=True):
        super(Model, self).infer_mode(recursive=recursive)
        cacher.invalidate(self, include_self=False, recursive=True)

    def _internal(self, *args, **kwargs):
        """
        Get internal IR information.
        TODO(yzhliu): we may consider APIs like following in the future.
        model.set_input(...)
        internal = model._internal()

        Parameters
        ----------
        args : mnm.ndarray
            The input data of the model.
        kwargs : named inputs
            Currently not supported.

        Returns
        -------
        record: _TraceRecord
            The traced record.
            Get relay function by record.func, parameters by record.named_params.
        """
        fwd_func = self.__fwd_train if self._BaseModel__is_train else self.__fwd_infer  # pylint: disable=no-member
        pyfunc = fwd_func.__wrapped__
        # TODO(hgt312): varargs and kwargs
        if args or kwargs:
            args = [self] + list(args)
        else:
            sig = inspect.signature(pyfunc)
            args = [self] + list(sig.parameters.keys())[1:]

        record = _get_trace_record(pyfunc, args, kwargs)
        return record


# pylint: disable=protected-access


def _get_attr_models_key_value(model):
    return get_named_attr(model, check=lambda x: isinstance(x, Model))


def _get_attr_models_value(model):
    return get_attr(model, check=lambda x: isinstance(x, Model))


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

    bfs([root_model],
        on_pop,
        on_next=_get_attr_models_value,
        recursive=recursive)


def _get_param_dict(root_model, *, prefix, recursive):
    model_prefix = {root_model: prefix}
    result = OrderedDict()

    def on_pop(model):
        prefix = model_prefix[model]
        if prefix != "":
            prefix = prefix + "."
        for name, item in _get_attr_params_key_value(model).items():
            result[prefix + name] = item
        for name, item in _get_attr_models_key_value(model).items():
            model_prefix[item] = prefix + name

    bfs([root_model],
        on_pop,
        on_next=_get_attr_models_value,
        recursive=recursive)
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

    bfs([root_model],
        on_pop,
        on_next=_get_attr_models_value,
        recursive=recursive)
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
