# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python package of RAF is a trivial extension to TVM's."""
import ctypes
import os
from raf._lib_utils import find_lib_path


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


def set_tvm_lib_path():
    """Set the TVM library path to use the RAF comppatible one."""
    lib_path = os.path.dirname(find_lib_path()[0])
    os.environ["TVM_LIBRARY_PATH"] = lib_path


# pylint: disable=unused-import, wrong-import-position

# Set the TVM library path before importing.
set_tvm_lib_path()

# Load TVM APIs.
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

# pylint: enable=unused-import, wrong-import-position

# Load RAF APIs.
_LIB, _LIB_NAME = _load_lib()
_APIS = _get_apis()


class TensorContainer(ctypes.Structure):
    # pylint: disable=too-few-public-methods
    """Python interface for NDArray::Container in tvm"""
    _fields_ = [("dltensor", _DLTensor), ("manager_ctx", ctypes.c_void_p)]
