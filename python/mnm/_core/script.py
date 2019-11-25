import contextlib
import inspect
import threading
from collections import namedtuple

import mnm
from mnm._core.core_utils import get_func_name
from mnm._lib import relay

from .model import Model
from .ndarray import Symbol

ScopeItem = namedtuple("ScopeItem", ["name", "mutation"])

class ScopeStack:
    storage = threading.local()

    @staticmethod
    @contextlib.contextmanager
    def scope(item):
        storage = ScopeStack.storage
        try:
            if not hasattr(storage, "stack"):
                storage.stack = []
            storage.stack.append(ScopeItem(name="script", mutation=[]))
            yield
        finally:
            assert hasattr(storage, "stack")
            popped = storage.stack.pop()
            assert popped.name == item

    @staticmethod
    def last() -> ScopeItem:
        storage = ScopeStack.storage

        if not getattr(storage, "stack", None):
            return ScopeItem(name=None, mutation=None)

        return storage.stack[-1]


def _script_switch(module):
    for name in module.__all__:
        setattr(mnm, name, getattr(module, name))


def script_mutate_attr(obj, attr_name, symbol):
    last = ScopeStack.last()
    assert last.name == "script"
    last.mutation.append((obj, attr_name, symbol))


def _script_bind_args(pyfunc, args, kwargs):
    sig = inspect.signature(pyfunc)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    params = []

    for name, value in list(bound_args.arguments.items())[1:]:
        if not isinstance(value, mnm.ndarray):
            raise NotImplementedError("We only support ndarray as inputs for now")
        symbol = Symbol.make_var(name_hint=name)
        bound_args.arguments[name] = symbol
        params.append(symbol._Symbol__expr)  # pylint: disable=protected-access

    for _, param in args[0].state().items():
        params.append(param._ndarray__handle._expr)  # pylint: disable=protected-access

    return bound_args, params


def _script_make_function(params, ret, mutation):
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
    ret = [x._Symbol__expr for x in ret]  # pylint: disable=protected-access
    ret = relay.Tuple(ret)
    ret = relay.Function(params=params, body=ret)

    return ret


def _script_get_cache(model, func_name):
    return model._Model__cache.get("script@" + func_name, None)  # pylint: disable=protected-access


def _script_run(pyfunc, args, kwargs):
    print("### Start scripting:", get_func_name(pyfunc))
    bound_args, params = _script_bind_args(pyfunc, args, kwargs)
    try:
        _script_switch(mnm._op.sym)  # pylint: disable=protected-access
        with ScopeStack.scope(item="script"):
            ret = pyfunc(*bound_args.args, **bound_args.kwargs)
            mutation = ScopeStack.last().mutation
    finally:
        _script_switch(mnm._op.imp)  # pylint: disable=protected-access
    ret = _script_make_function(params, ret, mutation)
    print("Result:", ret)

    return ret


def script_model(pyfunc):
    sig = inspect.signature(pyfunc)
    func_name = get_func_name(pyfunc)

    for _, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:  # var args
            raise NotImplementedError("Varargs are not supported for now")

        if param.kind == inspect.Parameter.VAR_KEYWORD: # kwargs
            raise NotImplementedError("Kwargs are not supported for now")

    def new_pyfunc(*args, **kwargs):
        if (not args) or not isinstance(args[0], Model):
            raise ValueError("The first argument of script_model is required to be Model itself")

        if ScopeStack.last().name != "script":
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
