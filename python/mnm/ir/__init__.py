"""IR related data structures and utils."""
from .._ffi.ir import AsText
from .._core.ir_ext import extended_var as var
from .._core.module import IRModule
from .._core import module
from .._lib import PassContext
from .serialization import save_json
from .constant import to_value, const
from .pass_manager import MNMSequential
from .scope_builder import ScopeBuilder
