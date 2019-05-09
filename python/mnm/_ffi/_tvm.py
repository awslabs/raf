"""TVM bridge. The Python package of MNM is a trivial extension to TVM's."""

import tvm
import tvm.relay

from tvm._ffi.base import decorate as _decorate

from tvm._ffi.base import register_error as _register_error
from tvm._ffi.base import TVMError as _TVMError
from tvm._ffi.base import py_str as _py_str
from tvm._ffi.base import numeric_types as _numeric_types
from tvm._ffi.base import string_types as _string_types

from tvm._ffi.function import _init_api
from tvm._ffi.function import Function as _Function
from tvm._ffi.function import get_global_func as _get_global_func
from tvm._ffi.function import register_func as _register_func
from tvm._ffi.function import list_global_func_names as _list_global_func_names

from tvm._ffi.ndarray import _NDArrayBase
from tvm._ffi.ndarray import register_extension as _register_extension
from tvm._ffi.ndarray import free_extension_handle as _free_extension_handle

from tvm._ffi.node import NodeBase as _NodeBase
from tvm._ffi.node import NodeGeneric as _NodeGeneric
from tvm._ffi.node import register_node as _register_node

from tvm._ffi.runtime_ctypes import TVMByteArray as _ByteArray
from tvm._ffi.runtime_ctypes import TVMType as _DLDataType
from tvm._ffi.runtime_ctypes import TVMContext as _DLContext
from tvm._ffi.runtime_ctypes import TVMArray as _DLTensor
from tvm._ffi.runtime_ctypes import TVMNDArrayContainer as _DLManagedTensor
