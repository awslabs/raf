# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python package of RAF is a trivial extension to TVM's.
"""
# pylint: disable=unused-import
import ctypes
import os
import readline
import sys

import tvm.topi as topi
import tvm
import tvm.relay as relay
from tvm._ffi.base import TVMError as _TVMError
from tvm._ffi.base import decorate as _decorate
from tvm._ffi.base import numeric_types as _numeric_types
from tvm._ffi.base import py_str as _py_str
from tvm._ffi.base import register_error as _register_error
from tvm._ffi.base import string_types as _string_types
from tvm.relay import Function as _Function
from tvm._ffi.registry import get_global_func as _get_global_func
from tvm._ffi.registry import list_global_func_names as _list_global_func_names
from tvm._ffi.registry import register_func as _register_func
from tvm._ffi.registry import register_extension as _register_extension
from tvm._ffi.registry import register_object as _register_object
from tvm.runtime.object_generic import ObjectBase as _NodeBase
from tvm.runtime.object import Object
from tvm.runtime.object_generic import ObjectGeneric as _NodeGeneric
from tvm._ffi.runtime_ctypes import TVMArray as _DLTensor
from tvm._ffi.runtime_ctypes import TVMByteArray as _ByteArray
from tvm._ffi.runtime_ctypes import Device as _DLDevice
from tvm._ffi.runtime_ctypes import TVMArrayHandle as _DLArrayHandle
from tvm.target import generic_func
from tvm.tir import FloatImm, IntImm, StringImm
from tvm.ir.container import Array
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
from tvm.runtime.ndarray import array as tvm_ndarray
from tvm.relay import op as _op
from tvm.relay.op import OpPattern, register_compute, register_pattern, strategy
from tvm.relay.op.op import (
    register_injective_schedule,
    register_broadcast_schedule,
    register_reduce_schedule,
)
from tvm.relay.op import op as _reg
from tvm.relay.dataflow_pattern import (
    DFPattern,
    is_var,
    is_expr,
    is_op,
    is_tuple,
    is_tuple_get_item,
    is_if,
    is_let,
    wildcard,
    has_type,
    has_dtype,
    has_shape,
    has_attr,
    dominates,
)
from tvm.contrib import random

# pylint: enable=unused-import


def find_lib_path(name=None, search_path=None):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.
    search_path : str
        Root path to search.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries
    """
    package_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..")

    dll_path = []

    if os.environ.get("RAF_LIBRARY_PATH", None):
        dll_path.append(os.environ["RAF_LIBRARY_PATH"])

    if sys.platform.startswith("linux") and os.environ.get("LD_LIBRARY_PATH", None):
        dll_path.extend([p.strip() for p in os.environ["LD_LIBRARY_PATH"].split(":")])
    elif sys.platform.startswith("darwin") and os.environ.get("DYLD_LIBRARY_PATH", None):
        dll_path.extend([p.strip() for p in os.environ["DYLD_LIBRARY_PATH"].split(":")])

    # Package data directory when pip installed
    dll_path.append(package_dir)
    # Pip lib directory
    dll_path.append(os.path.join(ffi_dir, ".."))
    # Default cmake build directory
    dll_path.append(os.path.join(source_dir, "build"))
    dll_path.append(os.path.join(source_dir, "build", "lib"))
    dll_path.append(os.path.join(source_dir, "build", "Release"))
    # Default make build directory
    dll_path.append(os.path.join(source_dir, "lib"))
    dll_path.append(install_lib_dir)

    dll_path = [os.path.realpath(x) for x in dll_path]

    if search_path is not None:
        if not isinstance(search_path, list):
            search_path = [search_path]
        dll_path.extend(search_path)

    if name is not None:
        if not isinstance(name, list):
            name = [name]
        lib_dll_path = [os.path.join(p, n) for n in name for p in dll_path]
    else:
        if sys.platform.startswith("win32"):
            lib_dll_path = [os.path.join(p, "libraf.dll") for p in dll_path] + [
                os.path.join(p, "raf.dll") for p in dll_path
            ]
        elif sys.platform.startswith("darwin"):
            lib_dll_path = [os.path.join(p, "libraf.dylib") for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, "libraf.so") for p in dll_path]

    lib_found = [p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_found:
        message = (
            "Cannot find the files.\n" + "List of candidates:\n" + str("\n".join(lib_dll_path))
        )
        raise RuntimeError(message)

    return lib_found


def _load_lib():
    lib_path = find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    lib.TVMGetLastError.restype = ctypes.c_char_p

    return lib, os.path.basename(lib_path[0])


def _get_apis():
    apis = {}

    for name in _list_global_func_names():
        if not name.startswith("raf."):
            continue
        func = _get_global_func(name)
        func.is_global = True
        func.__name__ = name
        func.__doc__ = "TVM PackedFunc %s. " % name
        apis[name] = func

    return apis


def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name, func in _APIS.items():
        if not name.startswith(prefix):
            continue
        name = name[len(prefix) + 1 :]

        if "." in name:
            continue
        setattr(module, func.__name__, func)


_LIB, _LIB_NAME = _load_lib()
_APIS = _get_apis()


class TensorContainer(ctypes.Structure):
    # pylint: disable=too-few-public-methods
    """Python interface for NDArray::Container in tvm"""
    _fields_ = [("dltensor", _DLTensor), ("manager_ctx", ctypes.c_void_p)]
