from collections import OrderedDict

from mnm._core import cacher
from mnm._core.core_utils import bfs, get_attr, get_named_attr
from mnm._core.ndarray import Parameter


class Model(cacher.Cacher):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.__is_train = True
        build, self.__fwd_train, self.__fwd_infer = _extract_methods(self)
        build(*args, **kwargs)
        cacher.enable(self)  # Cache is set up after the model is built

    def __call__(self, *args, **kwargs):
        forward = self.__fwd_train if self.__is_train else self.__fwd_infer
        return forward(*args, **kwargs)

    def train_mode(self, recursive=True):
        _set_is_train(self, value=True, recursive=recursive)
        cacher.invalidate(self, include_self=False, recursive=True)

    def infer_mode(self, recursive=True):
        _set_is_train(self, value=False, recursive=recursive)
        cacher.invalidate(self, include_self=False, recursive=True)

    def state(self, prefix="", recursive=True):
        return _get_param_dict(self, prefix=prefix, recursive=recursive)


# pylint: disable=protected-access


def _get_attr_models(model):
    return get_named_attr(model, check=lambda x: isinstance(x, Model))


def _get_attr_models_value(model):
    return get_attr(model, check=lambda x: isinstance(x, Model))


def _get_attr_params(model):
    return get_named_attr(model, check=lambda x: isinstance(x, Parameter))


def _get_attr_params_value(model):
    return get_attr(model, check=lambda x: isinstance(x, Parameter))


def _set_is_train(root_model, *, value, recursive):
    if not hasattr(root_model, "_Model__is_train"):
        return

    def on_pop(model):
        object.__setattr__(model, "_Model__is_train", value)
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
        for name, item in _get_attr_params(model).items():
            result[prefix + name] = item
        for name, item in _get_attr_models(model).items():
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
        raise NotImplementedError("Please implement build() method")
    if not fwd_train:
        raise NotImplementedError("Please implement forward() method")
    if not fwd_infer:
        fwd_infer = fwd_train
    return build, fwd_train, fwd_infer


# pylint: enable=protected-access
