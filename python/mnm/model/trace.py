import sys
from collections import OrderedDict, namedtuple

from mnm._core import cacher
from mnm._core.core_utils import get_bound_args, get_func_name
from mnm._core.global_scope import SCOPE
from mnm._core.ndarray import Symbol, ndarray
from mnm._ffi.pass_ import ExtractBinding, RenameVars
from mnm._lib import relay

from .model import Model


def trace_mutate_attr(obj, attr_name, symbol):
    last = _scope_last()
    assert last is not None
    assert last.name == "trace"
    last.mutate.append((obj, attr_name, symbol))


def trace(pyfunc):
    def new_pyfunc(*args, **kwargs):
        if _scope_last_name() == "trace":
            return pyfunc(*args, **kwargs)
        return _get_trace_record(pyfunc, args, kwargs)

    return new_pyfunc


_TraceRecord = namedtuple(
    "_TraceRecord",
    [
        "func",  # The relay.Function constructed by tracing
        "params",  # Model parameters that are extra inputs to the relay.Function
        # OrderedDict str -> ndarray/Parameter
        "struct",  # Structure of the outputs
        "mutations",  # [model, attr_name]
    ])


def _get_trace_record(pyfunc, args, kwargs):
    try:
        model = args[0]
        assert isinstance(model, Model)
    except:
        raise ValueError("@trace should only be applied to a model")
    func_name = get_func_name(pyfunc)
    record = cacher.get_cache(model, "trace@" + func_name, None)
    if record is not None:
        print("### Use cached trace:", func_name)
        return record
    print("### Start tracing:", func_name)
    record = _do_tracing(pyfunc, args, kwargs)
    cacher.set_cache(model, "trace@" + func_name, record)
    print("Result:", record.func)
    return record


def _do_tracing(pyfunc, args, kwargs):
    args, kwargs, input_named_vars = _bind_inputs(pyfunc, args, kwargs)
    try:
        _switch_imperative_symbolic("sym")
        with _scope(name="trace"):
            output = pyfunc(*args, **kwargs)
            mutations, mutate_symbols = _scope_last_mutate()
    finally:
        _switch_imperative_symbolic("imp")
    output, struct = _flatten_to_list(output)
    body = _extract_func_body(output + mutate_symbols)
    params = _get_used_params(args[0], body, input_named_vars)
    func = _make_func(body, input_named_vars, params)
    return _TraceRecord(func=func,
                        params=params,
                        struct=struct,
                        mutations=mutations)


def _make_func(body, input_named_vars, params):
    input_vars = list(input_named_vars.values())
    param_vars = [x._ndarray__handle for x in params.values()]  # pylint: disable=protected-access
    func = relay.Function(input_vars + param_vars, body=body)
    named_vars = dict()
    for name, var in input_named_vars.items():
        assert name not in named_vars
        named_vars[name] = var
    for name, param in params.items():
        handle = param._ndarray__handle  # pylint: disable=protected-access
        if name not in named_vars:
            named_vars[name] = handle
            continue
        suffix = 1
        while True:
            new_name = name + "_" + str(suffix)
            if new_name not in named_vars:
                named_vars[new_name] = handle
                break
            suffix += 1
    func = RenameVars(func, named_vars)
    return func


def _extract_func_body(outputs):
    body = [x._Symbol__handle for x in outputs]  # pylint: disable=protected-access
    if len(body) == 1:
        body = body[0]
    else:
        body = Symbol.make_tuple(body)._Symbol__handle  # pylint: disable=protected-access
    return ExtractBinding(body)


def _get_used_params(model, body, input_named_vars):
    free_vars = set(relay.analysis.free_vars(body))
    params = OrderedDict()
    for var in input_named_vars.values():
        if var in free_vars:
            free_vars.remove(var)
    for name, param in model.state().items():
        handle = param._ndarray__handle  # pylint: disable=protected-access
        if handle in free_vars:
            params[name] = param
            free_vars.remove(handle)
    if free_vars:
        raise ValueError("To ensure correctness, in tracing mode, "
                         "please do not use other ndarray/symbols "
                         "other than model's own")
    return params


def _bind_inputs(pyfunc, args, kwargs):
    # TODO(@junrushao1994): support varargs and kwargs
    bound_args = get_bound_args(pyfunc, args, kwargs)
    input_named_vars = OrderedDict()
    for name, value in list(bound_args.arguments.items())[1:]:
        if not isinstance(value, ndarray):
            raise NotImplementedError("Only ndarray is supported for now")
        symbol = Symbol.make_var(name_hint=name)
        bound_args.arguments[name] = symbol
        input_named_vars[name] = symbol._Symbol__handle  # pylint: disable=protected-access
    return bound_args.args, bound_args.kwargs, input_named_vars


def _flatten_to_list(a):
    if a is None:
        return [None], None
    if isinstance(a, (ndarray, Symbol)):
        return [a], ndarray
    if isinstance(a, (tuple, list)):
        flat_a = []
        struct = list()
        for item in a:
            flat_item, struct_item = _flatten_to_list(item)
            flat_a.extend(flat_item)
            struct.append((len(flat_item), struct_item))
        if isinstance(a, tuple):
            struct = tuple(struct)
        return flat_a, struct
    raise NotImplementedError("{} is not supported for now".format(type(a)))


def _unflatten_from_struct(a, struct):
    if struct is None:
        assert len(a) == 1 and a[0] is None
        return None
    if struct is ndarray:
        assert len(a) == 1
        return ndarray(a[0])
    if isinstance(struct, (tuple, list)):
        result = []
        for length, sub_struct in struct:
            sub_a = a[:length]
            a = a[length:]
            result.append(_unflatten_from_struct(sub_a, sub_struct))
        if isinstance(struct, tuple):
            result = tuple(result)
        return result
    raise NotImplementedError


def _switch_imperative_symbolic(target):
    mnm = sys.modules["mnm"]
    target = sys.modules["mnm._op." + target]
    for name in target.__all__:
        setattr(mnm, name, getattr(target, name))


# Scope support for tracing

_ScopeItem = namedtuple("_ScopeItem", ["name", "mutate"])


def _scope(name, mutate=None):
    if mutate is None:
        mutate = []
    return SCOPE.with_scope(_ScopeItem(name=name, mutate=mutate))


def _scope_last():
    return SCOPE.last(_ScopeItem, None)


def _scope_last_name():
    last = SCOPE.last(_ScopeItem, None)
    if last is None:
        return last
    return last.name


def _scope_last_mutate():
    last = SCOPE.last(_ScopeItem, None)
    if last is None:
        return []
    last = last.mutate
    mutations = []
    mutate_symbols = []
    for obj, attr, sym in last:
        mutations.append((obj, attr))
        mutate_symbols.append(sym)
    return mutations, mutate_symbols
