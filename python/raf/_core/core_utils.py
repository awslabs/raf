# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Utilities for core components."""
import inspect
import functools
from collections import deque, OrderedDict

from raf._lib import tvm
from raf._lib import _DLDevice
from raf._lib import _NodeBase as NodeBase  # pylint: disable=unused-import
from raf._lib import _register_object as _register_node

DEVICE_TYPE_MAP = {
    "llvm": 1,
    "stackvm": 1,
    "cpu": 1,
    "c": 1,
    "gpu": 2,
    "cuda": 2,
    "nvptx": 2,
    "cuda_host": 3,
    "cl": 4,
    "opencl": 4,
    "aocl": 5,
    "aocl_sw_emu": 5,
    "sdaccel": 6,
    "vulkan": 7,
    "metal": 8,
    "vpi": 9,
    "rocm": 10,
    "opengl": 11,
    "ext_dev": 12,
    "micro_dev": 13,
}


def register_node(type_key=None):
    assert isinstance(type_key, str)
    return _register_node(type_key)


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func

    return decorator


def _get_device_map():
    _str2ctx = {}
    for device_type, idx in DEVICE_TYPE_MAP.items():
        _str2ctx[device_type] = _DLDevice(device_type=idx, device_id=0)
        for device_id in range(128):
            name = f"{device_type}({device_id})"
            _str2ctx[name] = _DLDevice(device_type=idx, device_id=device_id)
    return _str2ctx


_STR2DEV = _get_device_map()


@tvm._ffi.register_func("raf._core.core_utils.dev2str")  # pylint: disable=protected-access
def dev2str(dev: _DLDevice) -> str:
    mask = [
        None,
        "cpu",
        "cuda",
        "cpu_pinned",
        "cl",
        "aocl",
        "sdaccel",
        "vulkan",
        "metal",
        "vpi",
        "rocm",
        "opengl",
    ]
    dev_type = int(dev.device_type)
    dev_id = int(dev.device_id)
    if dev_id == 0 and dev_type in (1, 3):
        return mask[dev_type]
    return mask[dev_type] + "(" + str(dev_id) + ")"


@tvm._ffi.register_func("raf._core.core_utils.str2dev")  # pylint: disable=protected-access
def str2dev(name: str) -> _DLDevice:
    return _STR2DEV[name]


def bfs(sources, on_pop, on_next, *, recursive=True):
    if not recursive:
        for item in sources:
            on_pop(item)
        return
    sources = list(sources)
    queue = deque(sources)
    visited = set(sources)
    while len(queue) > 0:
        model = queue.popleft()
        on_pop(model)
        for submodel in on_next(model):
            if submodel is not None and submodel not in visited:
                visited.add(submodel)
                queue.append(submodel)


def get_func_name(pyfunc):
    return pyfunc.__module__ + "$" + pyfunc.__qualname__


def get_bound_args(pyfunc, args, kwargs) -> inspect.BoundArguments:
    # pylint: disable=protected-access,too-many-locals
    sig = inspect.signature(pyfunc)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    # check if there's variable positional arguments
    var_pos_name = None
    var_keyword_name = None
    for name, _ in bound_args.arguments.items():
        if sig.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
            assert var_pos_name is None
            var_pos_name = name
        elif sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            assert var_keyword_name is None
            var_keyword_name = name

    if var_pos_name or var_keyword_name:
        # expand the variable positional arguments and update the signature
        new_params = []
        for name, param in sig.parameters.items():
            if name == var_pos_name:
                for i, arg in enumerate(bound_args.arguments[name]):
                    new_name = f"_p{i}"
                    assert new_name not in bound_args.arguments
                    bound_args.arguments[new_name] = arg
                    new_params.append(
                        inspect.Parameter(new_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    )
            elif name == var_keyword_name:
                for kw_name, kw_arg in bound_args.arguments[name].items():
                    assert kw_name not in bound_args.arguments
                    bound_args.arguments[kw_name] = kw_arg
                    new_params.append(
                        inspect.Parameter(kw_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    )
            else:
                new_params.append(param)

        if var_pos_name:
            del bound_args.arguments[var_pos_name]
        if var_keyword_name:
            del bound_args.arguments[var_keyword_name]
        sig = sig.replace(parameters=new_params)
        bound_args._signature = sig
    return bound_args


def get_attr(instance, *, name=None, check=None):
    single = False
    if name is None:
        name = dir(instance)
    elif not isinstance(name, (list, tuple)):
        single = True
        name = [name]
    ret = []
    for candidate in sorted(name):
        member = getattr(instance, candidate, None)
        if member is None:
            continue
        if (check is not None) and (not check(member)):
            continue
        ret.append(member)
    if single:
        if not ret:
            return None
        if len(ret) == 1:
            return ret[0]
        return ret
    return ret


def get_named_attr(instance, *, name=None, check=None):
    if name is None:
        name = dir(instance)
    elif not isinstance(name, (list, tuple)):
        name = [name]
    ret = OrderedDict()
    for candidate in sorted(name):
        member = getattr(instance, candidate, None)
        if member is None:
            continue
        if (check is not None) and (not check(member)):
            continue
        ret[candidate] = member
    return ret


def get_chained_attr(instance, names, default=None):
    for name in names:
        if not hasattr(instance, name):
            return default
        instance = getattr(instance, name)
    return instance


def with_signature(other, fmap):
    """
    Decorates a function (this) so that its signature becomes
    fmap(this.parameters.values(), other.parameters.values())

    Parameters
    ----------
    other: function
    the other function

    fmap: function
    fuse the signature of two functions
    """

    def decorator(this):
        @functools.wraps(this)
        def wrapped(*args, **kwargs):
            return this(*args, **kwargs)

        s_other = inspect.signature(other)
        s_this = inspect.signature(this)
        p_other = list(s_other.parameters.values())
        p_this = list(s_this.parameters.values())
        s_this = s_this.replace(parameters=fmap(p_this, p_other))
        wrapped.__signature__ = s_this
        return wrapped

    return decorator
