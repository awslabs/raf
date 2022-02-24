# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope builder instances."""

from tvm._ffi.base import string_types
from tvm.relay.expr import Var as RelayVar
from tvm.relay.scope_builder import ScopeBuilder as RelayScopeBuilder
from .._core.ir_ext import ExtendedVar, extended_var
from .._ffi.ir._make import Var as _Var


class ScopeBuilder(RelayScopeBuilder):
    """RAF scope builder, which is mostly derived from Relay's scope builder,
    but creates ExtendedVar instead of Var.
    """

    def let(self, var, value, may_share=None):  # pylint: disable=arguments-differ
        """Create a new let binding using ExtendedVar.

        Parameters
        ----------
        var: Union[str, Tuple[str, relay.Type], tvm.relay.Var, raf._core.ir_ext.ExtendedVar]
            The variable or name of variable.

        value: tvm.relay.Expr
            The value to be bound.

        may_share: Optional[raf._core.ir_ext.ExtendedVar]
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
