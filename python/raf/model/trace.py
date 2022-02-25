# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-function-docstring, protected-access
import functools
import sys
from collections import OrderedDict, namedtuple

from raf._core import cacher
from raf._core.core_utils import get_bound_args, get_func_name
from raf._core.global_scope import SCOPE
from raf._core.module import IRModule
from raf._core.ndarray import Symbol, ndarray
from raf._ffi.pass_ import ExtractBinding, RenameVars
from raf._ffi.model import RunModel
from raf._lib import relay, Array

_TraceRecord = namedtuple(
    "_TraceRecord",
    [
        "mod",  # The relay.Function constructed by tracing
        "named_params",  # Model parameters that are extra inputs to the relay.Function
        "o_struct",  # Structure of the outputs
        "mutations",  # [model, attr_name]
        "requires_grads",  # If input(s) of function requires gradient
    ],
)


def trace_mutate_attr(obj, attr_name, symbol):
    # pylint: disable=protected-access
    last = _scope_last()
    assert last is not None
    assert last.name == "trace"
    if (obj, attr_name) not in last.mutate:
        arr = obj.state()[attr_name]
        assert isinstance(arr, ndarray)
    else:
        arr = last.mutate[(obj, attr_name)][0]
    var = symbol._Symbol__handle
    source = arr._ndarray__handle
    assert isinstance(var, relay.Var)
    assert isinstance(source, relay.Var)
    last.mutate[(obj, attr_name)] = (arr, symbol)
    object.__setattr__(obj, attr_name, symbol)


def trace(pyfunc):
    @functools.wraps(pyfunc)
    def new_pyfunc(*args, **kwargs):
        if len(args) == 0 or not isinstance(args[0], cacher.Cacher):
            raise ValueError("Decorator trace should only be applied to a model")
        if _scope_last_name() == "trace":
            return pyfunc(*args, **kwargs)
        record = _get_trace_record(pyfunc, args, kwargs)
        bound_args = get_bound_args(pyfunc, args, kwargs)
        return _run_trace_record(record, bound_args.args, bound_args.kwargs)

    return new_pyfunc


# The logic of running a tracing record
def _run_trace_record(record, args, kwargs):
    func_inputs = _get_func_inputs(record, args[1:], kwargs)
    result = _unwrap(RunModel(record.mod, func_inputs))
    if not isinstance(result, list):
        result = [result]
    for obj, attr in reversed(record.mutations):
        object.__setattr__(obj, attr, result[-1])
        if attr in record.named_params.keys():
            record.named_params[attr] = result[-1]
        result.pop()
    return _unflatten_from_struct(result, record.o_struct)


def _get_handle_or_origin(arg, get_handle=True):
    if isinstance(arg, ndarray):
        return arg._ndarray__handle if get_handle else arg
    if isinstance(arg, Symbol):
        return arg._Symbol__handle if get_handle else arg
    if isinstance(arg, (tuple, list)):
        sym = Symbol.make_tuple([i._ndarray__handle for i in arg])
        return sym._Symbol__handle if get_handle else arg
    raise NotImplementedError("Not supported arg type: ", type(arg))


def _get_func_inputs(record, args, kwargs, get_handle=True):
    func = record.mod["main"]
    func_inputs = []
    for arg in args:
        func_inputs.append(_get_handle_or_origin(arg, get_handle))
    if kwargs:
        visited_params = set()
        for param_name in [p.name_hint for p in func.params]:
            if param_name not in kwargs:
                continue
            if param_name in visited_params:
                raise ValueError("Duplicated parameter: %s" % param_name)
            visited_params.add(param_name)
            val = kwargs[param_name]
            func_inputs.append(_get_handle_or_origin(val, get_handle))
    for param in record.named_params.values():
        handle = param._ndarray__handle if get_handle else param
        func_inputs.append(handle)
    if len(func_inputs) != len(func.params):
        raise ValueError(
            "Input size does not match main function params. %d v.s. %d."
            % (len(func_inputs), len(func.params))
        )
    return func_inputs


def _unwrap(result):
    if isinstance(result, relay.Var):
        return ndarray(result)
    if isinstance(result, Array):
        return [_unwrap(x) for x in result]
    return result


def _get_trace_record(pyfunc, args, kwargs):
    model = args[0]
    func_name = get_func_name(pyfunc)
    record = cacher.get_cache(model, "trace@" + func_name, None)
    if record is not None:
        return record
    record = _do_tracing(pyfunc, args, kwargs)
    cacher.set_cache(model, "trace@" + func_name, record)
    return record


# The logic of tracing
def _do_tracing(pyfunc, args, kwargs):
    # Step 1. switch input arguments to symbols
    named_inputs, args, kwargs = _symbolize_inputs(pyfunc, args, kwargs)
    # Step 2. run the traced function once
    try:
        _switch_imperative_symbolic("sym")
        with _scope(name="trace"):
            output = pyfunc(*args, **kwargs)
            mutations, mutate_arrays, mutate_symbols = _scope_last_mutate()
            for ((obj, attr), arr) in zip(mutations, mutate_arrays):
                object.__setattr__(obj, attr, arr)
    finally:
        _switch_imperative_symbolic("imp")
    # Step 3. flatten output to list, and keep record of its original structure
    output, o_struct = _flatten_to_list(output)
    # Step 4. and extra model parameters and finally make the relay.Func
    func, named_params = _make_func(args[0], named_inputs, output + mutate_symbols)
    mod = IRModule.from_expr(func)
    return _TraceRecord(
        mod=mod,
        named_params=named_params,
        o_struct=o_struct,
        mutations=mutations,
        requires_grads=[],
    )


def _symbolize_inputs(pyfunc, args, kwargs):
    # TODO(@junrushao1994): support varargs and kwargs
    def get_type(x):
        if isinstance(x, ndarray):
            return relay.TensorType(shape=x.shape, dtype=x.dtype)
        if isinstance(x, (tuple, list)):
            return relay.TupleType([get_type(i) for i in x])
        raise NotImplementedError("Type is not supported: ", type(x))

    bound_args = get_bound_args(pyfunc, args, kwargs)
    named_inputs = OrderedDict()
    for name, value in list(bound_args.arguments.items())[1:]:  # pylint: disable=unused-variable
        if isinstance(value, (tuple, list, ndarray)):
            bound_args.arguments[name] = named_inputs[name] = Symbol.make_var(
                name_hint=name, type_annotation=get_type(value)
            )
        elif isinstance(value, Symbol):
            bound_args.arguments[name] = named_inputs[name] = value
        else:
            raise NotImplementedError("Type is not supported: ", type(value))
    return named_inputs, bound_args.args, bound_args.kwargs


def _make_func(model, named_inputs, outputs):
    # Step 1. construct the function body
    body = _construct_func_body(outputs)
    # Step 2. scan the body and find out the model parameters used
    named_params = _get_used_params(model, body, named_inputs)
    # Step 3. construct the function using input vars and model parameters as inputs
    func = relay.Function(
        [x._Symbol__handle for x in named_inputs.values()]
        + [x._ndarray__handle for x in named_params.values()],
        body=body,
    )
    # Step 4. replace all variables in func, so that
    # 1) vars have better names
    # 2) get rid of referencing global binding table
    func = RenameVars(func, _get_named_vars(named_inputs, named_params))
    return func, named_params


def _construct_func_body(outputs):
    body = [x._Symbol__handle if isinstance(x, Symbol) else x._ndarray__handle for x in outputs]
    if len(body) == 1:
        body = body[0]
    else:
        body = Symbol.make_tuple(body)._Symbol__handle
    return ExtractBinding(body, [])


def _get_used_params(model, body, named_inputs):
    free_vars = set(relay.analysis.free_vars(body))
    named_params = OrderedDict()
    for sym in named_inputs.values():
        handle = sym._Symbol__handle
        if handle in free_vars:
            free_vars.remove(handle)
    for name, param in model.state().items():
        handle = param._ndarray__handle
        if handle in free_vars:
            named_params[name] = param
            free_vars.remove(handle)
    if free_vars:
        raise ValueError(
            "To ensure correctness, in tracing mode, "
            "please do not use other ndarray/symbols "
            "other than model's own"
        )
    return named_params


def _get_named_vars(named_inputs, named_params):
    named_vars = dict()
    for name, var in named_inputs.items():
        handle = var._Symbol__handle
        assert name not in named_vars
        named_vars[name] = handle
    for name, param in named_params.items():
        handle = param._ndarray__handle
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
    if isinstance(a, (ndarray)):
        return [a], ndarray
    if isinstance(a, (Symbol)):
        return [a], Symbol
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
    if struct is Symbol:
        length = len(a)
        if length == 1:
            return ndarray(a[0])
        out_array = []
        for i in range(length):
            out_array.append(ndarray(a[i]))
        return tuple(out_array)
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
    raf = sys.modules["raf"]
    target = sys.modules["raf._op." + target]
    for name in target.__all__:
        setattr(raf, name, getattr(target, name))


# Scope support for tracing

_ScopeItem = namedtuple("_ScopeItem", ["name", "mutate"])


def _scope(name, mutate=None):
    if mutate is None:
        # mutate is a dictionary of the following form:
        # mutate[(obj, attr_name)] = (arr, new_symbol)
        # where the obj is typically a Model object
        # attr_name is the name of an obj's attribute
        # obj.attr_name refers to the mutable param.
        # arr is the initial value of obj.attr_name
        # new_symbol is the current value of obj.attr_name
        # (because obj.attr_name is mutable, its value varies by time)
        mutate = {}
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
    mutate_arrays = []
    for ((obj, attr), (arr, sym)) in last.items():
        mutations.append((obj, attr))
        mutate_arrays.append(arr)
        mutate_symbols.append(sym)
    return mutations, mutate_arrays, mutate_symbols
