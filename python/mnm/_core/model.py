import weakref
from collections import OrderedDict, deque

from .ndarray import Parameter


def _get_callable_impl(instance, method_name):
    instance_method = getattr(instance, method_name, None)

    return instance_method if callable(instance_method) else None


class Model:

    def __init__(self, *args, **kwargs):
        self.__parents = weakref.WeakSet()
        self.__models = OrderedDict()
        self.__states = OrderedDict()
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
        self.__cache = dict()

    def __call__(self, *args, **kwargs):
        forward = self.__forward_train if self.__is_train else self.__forward_infer

        return forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Model):
            value._Model__parents.add(self)
            self.__models[name] = value
            self.__invalidate_cache()
        elif isinstance(value, Parameter):
            value._Parameter__parents.add(self)
            self.__states[name] = value
            self.__invalidate_cache()
        elif name == "_Model__is_train":
            if hasattr(self, name) and getattr(self, name) != value:
                self.__invalidate_cache()
        super().__setattr__(name, value)


    def __invalidate_cache(self):
        if not hasattr(self, "_Model__cache"):
            return
        self.__cache = {}
        print("Invalidating cache of", self)

        for parent in list(self.__parents.data):
            parent = parent()

            if parent is None:
                continue
            parent.__invalidate_cache()  # pylint: disable=protected-access

    def __delattr__(self, name):
        value = getattr(self, name)

        if isinstance(value, Model):
            value._Model__parents.remove(self)
            del self.__models[name]
        elif isinstance(value, Parameter):
            value._Parameter__parents.remove(self)
            del self.__states[name]
        super().__delattr__(name)

    def train_mode(self, recursive=True):
        self.__is_train = True

        if recursive:
            for name in dir(self):
                submodel = getattr(self, name)

                if isinstance(submodel, Model):
                    submodel.train_mode(recursive=True)

    def infer_mode(self, recursive=True):
        self.__is_train = False

        if recursive:
            for name in dir(self):
                submodel = getattr(self, name)

                if isinstance(submodel, Model):
                    submodel.infer_mode(recursive=True)

    def state(self, recursive=True, prefix=""):
        queue = deque([(prefix, self)])
        visited = {self}
        result = OrderedDict()

        while len(queue) > 0:
            prefix, model = queue.popleft()

            if prefix != "":
                prefix = prefix + "."

            for name, item in model.__states.items():  # pylint: disable=protected-access
                result[prefix + name] = item

            if not recursive:
                break

            for name, item in model.__models.items():  # pylint: disable=protected-access
                if item not in visited:
                    visited.add(item)
                    queue.append((prefix + name, item))

        return result
