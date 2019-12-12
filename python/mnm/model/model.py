import sys
import contextlib
import inspect
import threading
import weakref
from collections import OrderedDict, deque, namedtuple

from mnm._core.core_utils import get_func_name
from mnm._core.ndarray import Parameter, Symbol, ndarray
from mnm._ffi.binding import ExtractLetList
from mnm._ffi.pass_ import UnbindConstants


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
            param.requires_grad = value

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


class _ScopeStack:
    _ScopeItem = namedtuple("ScopeItem", ["name", "mutation"])
    storage = threading.local()

    @staticmethod
    @contextlib.contextmanager
    def scope(item):
        storage = _ScopeStack.storage
        try:
            if not hasattr(storage, "stack"):
                storage.stack = []
            storage.stack.append(
                _ScopeStack._ScopeItem(name="script", mutation=[]))
            yield
        finally:
            assert hasattr(storage, "stack")
            popped = storage.stack.pop()
            assert popped.name == item

    @staticmethod
    def last():
        storage = _ScopeStack.storage
        if not getattr(storage, "stack", None):
            return _ScopeStack._ScopeItem(name=None, mutation=None)
        return storage.stack[-1]


def _script_switch(module):
    mnm = sys.modules["mnm"]
    for name in module.__all__:
        setattr(mnm, name, getattr(module, name))


def _script_bind_args(pyfunc, args, kwargs):
    sig = inspect.signature(pyfunc)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    param_dict = OrderedDict()
    for name, value in list(bound_args.arguments.items())[1:]:
        if not isinstance(value, ndarray):
            raise NotImplementedError(
                "We only support ndarray as inputs for now")
        symbol = Symbol.make_var(name_hint=name)
        bound_args.arguments[name] = symbol
        param_dict[name] = symbol._Symbol__handle  # pylint: disable=protected-access
    for name, param in args[0].state().items():
        param_dict[name] = param._ndarray__handle  # pylint: disable=protected-access
    return bound_args, param_dict


def _script_make_function(param_dict, ret, mutation):
    # TODO(@junrushao1994): handle ndarrays
    # TODO(@junrushao1994): handle nested results
    if ret is None:
        ret = []
    elif isinstance(ret, Symbol):
        ret = [ret]
    if isinstance(ret, tuple):
        ret = list(ret)
    for _, _, symbol in mutation:
        ret.append(symbol)
    ret = [x._Symbol__handle for x in ret]  # pylint: disable=protected-access
    ret = Symbol.make_tuple(ret)
    ret = ExtractLetList(ret._Symbol__handle, list(param_dict.values()))  # pylint: disable=protected-access
    ret = UnbindConstants(ret, dict(param_dict))
    return ret


def _script_get_cache(model, func_name):
    return model._Model__cache.get("script@" + func_name, None)  # pylint: disable=protected-access


def _script_run(pyfunc, args, kwargs):
    mnm = sys.modules["mnm"]
    print("### Start scripting:", get_func_name(pyfunc))
    bound_args, param_dict = _script_bind_args(pyfunc, args, kwargs)
    try:
        _script_switch(mnm._op.sym)  # pylint: disable=protected-access
        with _ScopeStack.scope(item="script"):
            ret = pyfunc(*bound_args.args, **bound_args.kwargs)
            mutation = _ScopeStack.last().mutation
    finally:
        _script_switch(mnm._op.imp)  # pylint: disable=protected-access
    ret = _script_make_function(param_dict, ret, mutation)
    print("Result:", ret)
    return ret


def script_model(pyfunc):
    sig = inspect.signature(pyfunc)
    func_name = get_func_name(pyfunc)
    for _, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:  # var args
            raise NotImplementedError("Varargs are not supported for now")
        if param.kind == inspect.Parameter.VAR_KEYWORD:  # kwargs
            raise NotImplementedError("Kwargs are not supported for now")

    def new_pyfunc(*args, **kwargs):
        if (not args) or not isinstance(args[0], Model):
            raise ValueError(
                "The first argument of script_model is required to be Model itself"
            )
        if _ScopeStack.last().name != "script":
            model = args[0]
            ret = _script_get_cache(model, func_name)
            if ret is not None:
                print("### Use cached script:", get_func_name(pyfunc))
            else:
                ret = _script_run(pyfunc, args, kwargs)
                model._Model__cache["script@" + func_name] = ret  # pylint: disable=protected-access
        else:
            ret = pyfunc(*args, **kwargs)
        return ret

    return new_pyfunc


def script_mutate_attr(obj, attr_name, symbol):
    last = _ScopeStack.last()
    assert last.name == "script"
    last.mutation.append((obj, attr_name, symbol))
