# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-import
"""The RAF Pattern Language and tooling."""
from raf._lib import (
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
from raf._ffi.ir.dataflow_pattern import is_constant
from raf._ffi.pass_ import dataflow_pattern_match as match
from raf._ffi.pass_ import dataflow_pattern_rewrite as rewrite
from raf._ffi.pass_ import dataflow_pattern_partition as partition
