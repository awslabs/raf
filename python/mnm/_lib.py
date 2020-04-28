"""The Python package of MNM is a trivial extension to TVM's.
"""
# pylint: disable=unused-import
import ctypes
import os
import readline
import sys

import topi
import tvm
import tvm.relay as relay
from tvm._ffi.base import TVMError as _TVMError
from tvm._ffi.base import decorate as _decorate
from tvm._ffi.base import numeric_types as _numeric_types
from tvm._ffi.base import py_str as _py_str
from tvm._ffi.base import register_error as _register_error
from tvm._ffi.base import string_types as _string_types
from tvm._ffi.function import Function as _Function
from tvm._ffi.function import get_global_func as _get_global_func
from tvm._ffi.function import list_global_func_names as _list_global_func_names
from tvm._ffi.function import register_func as _register_func
from tvm._ffi.ndarray import _NDArrayBase
from tvm._ffi.ndarray import register_extension as _register_extension
from tvm._ffi.node import NodeBase as _NodeBase
from tvm._ffi.node import NodeGeneric as _NodeGeneric
from tvm._ffi.node import register_node as _register_node
from tvm._ffi.runtime_ctypes import TVMArray as _DLTensor
from tvm._ffi.runtime_ctypes import TVMByteArray as _ByteArray
from tvm._ffi.runtime_ctypes import TVMContext as _DLContext
# from tvm._ffi.runtime_ctypes import TVMNDArrayContainer as _DLManagedTensor
from tvm._ffi.runtime_ctypes import TVMType as _DLDataType
from tvm.container import Array
from tvm.expr import FloatImm, IntImm, StringImm
from tvm.make import node as _make_node
from tvm.ndarray import array as tvm_ndarray
from tvm.relay.op import (OpPattern, register_compute, register_pattern,
                          register_schedule)

# pylint: enable=unused-import

def find_lib_path(name=None, search_path=None):
    ffi_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(ffi_dir, "..", "..")
    install_lib_dir = os.path.join(ffi_dir, "..", "..", "..")

    dll_path = []

    if os.environ.get('MNM_LIBRARY_PATH', None):
        dll_path.append(os.environ['MNM_LIBRARY_PATH'])

    if sys.platform.startswith('linux') and os.environ.get(
            'LD_LIBRARY_PATH', None):
        dll_path.extend(
            [p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])
    elif sys.platform.startswith('darwin') and os.environ.get(
            'DYLD_LIBRARY_PATH', None):
        dll_path.extend(
            [p.strip() for p in os.environ['DYLD_LIBRARY_PATH'].split(":")])

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
        if sys.platform.startswith('win32'):
            lib_dll_path = [os.path.join(p, 'libmnm.dll') for p in dll_path] +\
                           [os.path.join(p, 'mnm.dll') for p in dll_path]
        elif sys.platform.startswith('darwin'):
            lib_dll_path = [os.path.join(p, 'libmnm.dylib') for p in dll_path]
        else:
            lib_dll_path = [os.path.join(p, 'libmnm.so') for p in dll_path]

    lib_found = [
        p for p in lib_dll_path if os.path.exists(p) and os.path.isfile(p)
    ]

    if not lib_found:
        message = ('Cannot find the files.\n' + 'List of candidates:\n' +
                   str('\n'.join(lib_dll_path)))
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
        if not name.startswith("mnm."):
            continue
        func = _get_global_func(name)
        func.is_global = True
        func.__name__ = name
        func.__doc__ = ("TVM PackedFunc %s. " % name)
        apis[name] = func

    return apis


def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name, func in _APIS.items():
        if not name.startswith(prefix):
            continue
        name = name[len(prefix) + 1:]

        if "." in name:
            continue
        setattr(module, func.__name__, func)


_LIB, _LIB_NAME = _load_lib()
_APIS = _get_apis()

# pylint: disable=invalid-name
nd_get_manager_ctx = tvm.get_global_func("mnm.tensor.nd_get_manager_ctx")
# pylint: enable=invalid-name

@tvm.register_object("mnm.tensor.Tensor")
class Tensor(tvm.Object):
    # pylint: disable=too-few-public-methods
    """Subclassing TVM's NDArray infrastructure."""
    @property
    def _tvm_handle(self):
        return self.handle.value

    @property
    def manager_ctx(self):
        return nd_get_manager_ctx(self)
