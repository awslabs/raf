import weakref
from collections import OrderedDict, deque

from .ndarray import Parameter


def _get_callable_impl(instance, method_name):
    instance_method = getattr(instance, method_name, None)
    return instance_method if callable(instance_method) else None


def _search_model(model, on_pop, on_next):
    queue = deque([model])
    visited = {model}
    while len(queue) > 0:
        model = queue.popleft()
        on_pop(model)
        for submodel in on_next(model):
            if submodel is not None and submodel not in visited:
                visited.add(submodel)
                queue.append(submodel)


# pylint: disable=protected-access
def _set_is_train(self, value, recursive):
    if not hasattr(self, "_Model__is_train"):
        return
    if not recursive:
        self._Model__is_train = value
        return

    def set_mode(model):
        model._Model__is_train = value
        for param in model._Model__params.values():
            param._Parameter__switch_mode(value)

    def get_next(model):
        return model._Model__models.values()

    _search_model(self, set_mode, get_next)
    self._Model__invalidate_cache(recursive=True)


# pylint: enable=protected-access


class Model:
    def __init__(self, *args, **kwargs):
        self.__parents = weakref.WeakSet()
        self.__models = OrderedDict()
        self.__params = OrderedDict()
        self.__is_train = True
        # Get methods
        build = _get_callable_impl(self, "build")
        forward_train = _get_callable_impl(self, "forward")
        forward_infer = _get_callable_impl(self, "forward_infer")
        # Check if methods are properly overridden
        if build is None:
            raise NotImplementedError("Please implement build() method")
        if forward_train is None:
            raise NotImplementedError("Please implement forward() method")
        if forward_infer is None:
            forward_infer = forward_train
        # Build the model
        build(*args, **kwargs)  # pylint: disable=not-callable
        self.__forward_train = forward_train
        self.__forward_infer = forward_infer
        # Cache is set up after the model is built
        self.__cache = dict()

    def __call__(self, *args, **kwargs):
        forward = self.__forward_train if self.__is_train else self.__forward_infer
        return forward(*args, **kwargs)

    def __delattr__(self, name):
        value = getattr(self, name)
        if isinstance(value, Model):
            value._Model__parents.remove(self)
            del self.__models[name]
            self.__invalidate_cache()
        elif isinstance(value, Parameter):
            value._Parameter__parents.remove(self)
            del self.__params[name]
            self.__invalidate_cache()
        super().__delattr__(name)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            delattr(self, name)
        if isinstance(value, Model):
            value._Model__parents.add(self)
            self.__models[name] = value
            self.__invalidate_cache()
        elif isinstance(value, Parameter):
            value._Parameter__parents.add(self)
            self.__params[name] = value
            self.__invalidate_cache()
        super().__setattr__(name, value)

    def __invalidate_cache(self, recursive=True):
        if not hasattr(self, "_Model__cache"):
            return
        if not recursive:
            self.__cache = {}
            return
        # pylint: disable=protected-access
        def invalidate(model):
            model._Model__invalidate_cache(recursive=False)

        def get_next(model):
            return [x() for x in list(model._Model__parents.data)]

        # pylint: enable=protected-access

        _search_model(self, invalidate, get_next)

    def train_mode(self, recursive=True):
        _set_is_train(self, True, recursive=recursive)

    def infer_mode(self, recursive=True):
        _set_is_train(self, False, recursive=recursive)

    def state(self, recursive=True, prefix=""):
        queue = deque([(prefix, self)])
        visited = {self}
        result = OrderedDict()
        while len(queue) > 0:
            prefix, model = queue.popleft()
            if prefix != "":
                prefix = prefix + "."
            for name, item in model.__params.items():  # pylint: disable=protected-access
                result[prefix + name] = item
            if not recursive:
                break
            for name, item in model.__models.items():  # pylint: disable=protected-access
                if item not in visited:
                    visited.add(item)
                    queue.append((prefix + name, item))
        return result
