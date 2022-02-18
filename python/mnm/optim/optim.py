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

"""Traced Optimizers"""
from mnm.frontend.model import _get_func_output_var
from mnm.ir import MNMSequential
from .. import distributed as dist
from .._core.ndarray import Symbol, get_symbol_handle
from .._core.value import NoGradValue, Value
from .._core.ir_ext import ExtendedVar
from ..model.trace import _get_func_inputs
from ..model import Model, trace
from .._ffi.pass_ import AutoDiff, InlineBackward, Substitute, InferType, FoldConstant
from .._ffi.pass_ import DeadCodeElimination, AutoDataParallel
from .._ffi.binding import BindSymbol
from .._lib import tvm


def calc_dy(dy, record):
    """relay function returns output + mutation. In backward, mutation needs empty gradient."""
    # pylint: disable=protected-access
    mod = InferType()(record.mod)
    ret_var = _get_func_output_var(mod["main"])
    dout = [get_symbol_handle(dy)]
    if isinstance(ret_var.checked_type, tvm.relay.TupleType):
        dout.extend(
            [
                Value.as_const_expr(NoGradValue())
                for i in range(len(ret_var.checked_type.fields) - 1)
            ]
        )
    return Symbol.make_tuple(dout) if len(dout) > 1 else dy


def inline(func, inputs):
    """inline execution func(*inputs)"""
    assert len(func.params) == len(inputs)
    vmap = dict(zip(func.params, inputs))

    def evaluate(body):
        if isinstance(body, tvm.relay.Var):
            return vmap[body]
        if isinstance(body, tvm.relay.Let):
            var = BindSymbol(Substitute(body.value, vmap), "", None)
            assert body.var not in vmap
            vmap[body.var] = var
            may_share = ExtendedVar(body.var).may_share
            if may_share is not None:
                assert may_share in vmap
                may_share = vmap[may_share]
            return evaluate(body.body)
        raise NotImplementedError("Not supported type: ", type(body))

    return Symbol.from_expr(evaluate(func.body))


def with_autodiff(model):
    """create a new model by apply autodiff to the input"""

    class AutoDiffWrapper(Model):
        """AutoDiff model

        Parameters
        ----------
        model: the forward model
        """

        def build(self, model):
            # pylint: disable=attribute-defined-outside-init, missing-function-docstring
            self.model = model

        @trace
        def forward(self, dy, *args, **kwargs):
            # pylint: disable=protected-access, missing-function-docstring
            record = self.model._internal(*args, **kwargs)
            dy = calc_dy(dy, record)
            mod = record.mod
            passes = [InferType(), AutoDiff(record.requires_grads)]
            if dist.get_context().enable_data_parallel:
                # TODO: Refactor AutoDataParallel to let it work on the IR after InlineBackward.
                passes += [AutoDataParallel()]
            passes += [InferType(), FoldConstant(), DeadCodeElimination(), InlineBackward()]
            seq = MNMSequential(passes)
            mod = seq(mod)
            inputs = _get_func_inputs(record, args, kwargs)
            inputs = inputs + [get_symbol_handle(dy)]
            out = inline(mod["main"], inputs)
            y = out[0]
            dxs = out[1]
            return y, dxs

    return AutoDiffWrapper(model)
