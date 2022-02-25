# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""IR related data structures and utils."""
from .._ffi.ir import AsText
from .._core.ir_ext import extended_var as var
from .._core.module import IRModule
from .._core import module
from .._lib import PassContext
from . import dataflow_pattern
from . import op
from .serialization import save_json, load_json
from .constant import to_value, const
from .pass_manager import RAFSequential
from .scope_builder import ScopeBuilder
from .anf_builder import ANFBuilder
