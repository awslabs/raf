import inspect
import functools
import sys
from collections import OrderedDict, namedtuple

from mnm._core import cacher
from mnm._core.core_utils import get_bound_args, get_func_name
from mnm._core.global_scope import SCOPE
from mnm._core.ndarray import Symbol, ndarray
from mnm._ffi.pass_ import ExtractBinding, RenameVars
from mnm._ffi.model import RunModel
from mnm._lib import relay, Array

_TraceRecord = namedtuple(
    "_TraceRecord",
    [
        "func",  # The relay.Function constructed by tracing
        "named_params",  # Model parameters that are extra inputs to the relay.Function
        "o_struct",  # Structure of the outputs
        "mutations",  # [model, attr_name]
    ])


def trace_mutate_attr(obj, attr_name, symbol):
    last = _scope_last()
    assert last is not None
    assert last.name == "trace"
    last.mutate.append((obj, attr_name, symbol))


def trace(pyfunc):
    @functools.wraps(pyfunc)
    def new_pyfunc(*args, **kwargs):
        if len(args) == 0 or not isinstance(args[0], cacher.Cacher):
            raise ValueError(
                "Decorator trace should only be applied to a model")
        if _scope_last_name() == "trace":
            return pyfunc(*args, **kwargs)
        record = _get_trace_record(pyfunc, args, kwargs)
        bound_args = get_bound_args(pyfunc, args, kwargs)
        return _run_trace_record(record, bound_args.args, bound_args.kwargs)

    return new_pyfunc


def _get_traced_func(model, traced_func, *args, **kwargs):
    # TODO(hgt312): varargs and kwargs
    pyfunc = traced_func.__wrapped__
    func_name = get_func_name(pyfunc)
    record = cacher.get_cache(model, "trace@" + func_name, None)
    if record is not None:
        return record.func
    if args or kwargs:
        args = [model] + args
    else:
        sig = inspect.signature(pyfunc)
        args = [model] + list(sig.parameters.keys())[1:]
    record = _do_tracing(pyfunc, args, {})
    cacher.set_cache(model, "trace@" + func_name, record)
    return record.func


# The logic of running a tracing record


def _run_trace_record(record, args, kwargs):
    if kwargs:
        # TODO(@junrushao1994): implement it
        raise NotImplementedError("keyword arguments not supported yet.")
    func_inputs = []
    for arg in args[1:]:
        if not isinstance(arg, ndarray):
            raise NotImplementedError("Only ndarray is supported for now")
        handle = arg._ndarray__handle  # pylint: disable=protected-access
        func_inputs.append(handle)
    for param in record.named_params.values():
        handle = param._ndarray__handle  # pylint: disable=protected-access
        func_inputs.append(handle)

    result = _unwrap(RunModel(record.func, func_inputs))
    if not isinstance(result, list):
        result = [result]
    for obj, attr in reversed(record.mutations):
        object.__setattr__(obj, attr, result[-1])
        result.pop()
    return _unflatten_from_struct(result, record.o_struct)


def _unwrap(result):
    if isinstance(result, relay.Var):
        return ndarray(result)
    if isinstance(result, Array):
        return [_unwrap(x) for x in result]
    raise NotImplementedError(type(result))


# The logic of tracing


def _get_trace_record(pyfunc, args, kwargs):
    model = args[0]
    func_name = get_func_name(pyfunc)
    record = cacher.get_cache(model, "trace@" + func_name, None)
    if record is not None:
        return record
    record = _do_tracing(pyfunc, args, kwargs)
    # print(record.func)
    cacher.set_cache(model, "trace@" + func_name, record)
    return record


def _do_tracing(pyfunc, args, kwargs):
    # Step 1. switch input arguments to symbols
    named_inputs, args, kwargs = _symbolize_inputs(pyfunc, args, kwargs)
    # Step 2. run the traced function once
    try:
        _switch_imperative_symbolic("sym")
        with _scope(name="trace"):
            output = pyfunc(*args, **kwargs)
            mutations, mutate_symbols = _scope_last_mutate()
    finally:
        _switch_imperative_symbolic("imp")
    # Step 3. flatten output to list, and keep record of its original structure
    output, o_struct = _flatten_to_list(output)
    # Step 4. and extra model parameters and finally make the relay.Func
    func, named_params = _make_func(args[0], named_inputs,
                                    output + mutate_symbols)
    return _TraceRecord(func=func,
                        named_params=named_params,
                        o_struct=o_struct,
                        mutations=mutations)


def _symbolize_inputs(pyfunc, args, kwargs):
    # TODO(@junrushao1994): support varargs and kwargs
    bound_args = get_bound_args(pyfunc, args, kwargs)
    named_inputs = OrderedDict()
    for name, value in list(bound_args.arguments.items())[1:]:  # pylint: disable=unused-variable
        ## comment the check to legalize fake inputs
        ## the same check is also in `_run_trace_record`
        # if not isinstance(value, ndarray):
        #     raise NotImplementedError("Only ndarray is supported for now")
        bound_args.arguments[name] = \
                named_inputs[name] = \
                Symbol.make_var(name_hint=name)
    return named_inputs, bound_args.args, bound_args.kwargs


def _make_func(model, named_inputs, outputs):
    # Step 1. construct the function body
    body = _construct_func_body(outputs)
    # Step 2. scan the body and find out the model parameters used
    named_params = _get_used_params(model, body, named_inputs)
    # Step 3. construct the function using input vars and model parameters as inputs
    func = relay.Function(
        [x._Symbol__handle for x in named_inputs.values()] +  # pylint: disable=protected-access
        [x._ndarray__handle for x in named_params.values()],  # pylint: disable=protected-access
        body=body)
    # Step 4. replace all variables in func, so that
    # 1) vars have better names
    # 2) get rid of referencing global binding table
    func = RenameVars(func, _get_named_vars(named_inputs, named_params))
    return func, named_params


def _construct_func_body(outputs):
    body = [x._Symbol__handle for x in outputs]  # pylint: disable=protected-access
    if len(body) == 1:
        body = body[0]
    else:
        body = Symbol.make_tuple(body)._Symbol__handle  # pylint: disable=protected-access
    return ExtractBinding(body)


def _get_used_params(model, body, named_inputs):
    free_vars = set(relay.analysis.free_vars(body))
    named_params = OrderedDict()
    for sym in named_inputs.values():
        handle = sym._Symbol__handle  # pylint: disable=protected-access
        if handle in free_vars:
            free_vars.remove(handle)
    for name, param in model.state().items():
        handle = param._ndarray__handle  # pylint: disable=protected-access
        if handle in free_vars:
            named_params[name] = param
            free_vars.remove(handle)
    if free_vars:
        raise ValueError("To ensure correctness, in tracing mode, "
                         "please do not use other ndarray/symbols "
                         "other than model's own")
    return named_params


def _get_named_vars(named_inputs, named_params):
    named_vars = dict()
    for name, var in named_inputs.items():
        handle = var._Symbol__handle  # pylint: disable=protected-access
        assert name not in named_vars
        named_vars[name] = handle
    for name, param in named_params.items():
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
    return named_vars


# Manipulate structures


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


# A simple user-facing switcher


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
