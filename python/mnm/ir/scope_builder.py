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

"""Scope builder instances."""

from tvm._ffi.base import string_types
from tvm.relay.expr import Var as RelayVar
from tvm.relay.scope_builder import ScopeBuilder as RelayScopeBuilder
from .._core.ir_ext import ExtendedVar, extended_var
from .._ffi.ir._make import Var as _Var


class ScopeBuilder(RelayScopeBuilder):
    """Meta scope builder, which is mostly derived from Relay's scope builder,
    but creates ExtendedVar instead of Var.
    """

    def let(self, var, value, may_share=None):  # pylint: disable=arguments-differ
        """Create a new let binding using ExtendedVar.

        Parameters
        ----------
        var: Union[str, Tuple[str, relay.Type], tvm.relay.Var, mnm._core.ir_ext.ExtendedVar]
            The variable or name of variable.

        value: tvm.relay.Expr
            The value to be bound.

        may_share: Optional[mnm._core.ir_ext.ExtendedVar]
            The shared variable.
        """
        if isinstance(var, (tuple, list)):
            if len(var) > 2:
                raise ValueError("Expect var to be Tuple[str, relay.Type]")
            var = extended_var(*var, may_share=may_share)
        elif isinstance(var, string_types):
            var = extended_var(var, may_share=may_share)
        elif isinstance(var, RelayVar):
            var = ExtendedVar(_Var(var, may_share=may_share))

        assert isinstance(var, ExtendedVar)
        self._bindings[-1].append((var, value))
        return var
