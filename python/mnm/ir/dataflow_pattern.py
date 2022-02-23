# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=unused-import
"""The Meta Pattern Language and tooling."""
from mnm._lib import (
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
from mnm._ffi.ir.dataflow_pattern import is_constant
from mnm._ffi.pass_ import dataflow_pattern_match as match
from mnm._ffi.pass_ import dataflow_pattern_rewrite as rewrite
from mnm._ffi.pass_ import dataflow_pattern_partition as partition
