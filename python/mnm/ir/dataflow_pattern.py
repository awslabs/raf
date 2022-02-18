# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
