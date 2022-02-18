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

"""A data parallel wrapper. Assuming the input model includes forward/backward computations, and
output 1) forward result as well as 2) all calculated gradients.
This wrapper enables data parallelism to train the model on multiple distributed devices.
The methodologies includes:
1. Input data parallelism: Assuming the input training data (e.g., mini-batches) are distributed
   to each device, we use all-reduce to aggregate partial gradients calculated by each device.
2. ZeRO optimizations (https://arxiv.org/abs/1910.02054):
   2.1 (ZeRO-1): Partition the gradients outputed by the given model, so that the later wrapped
                 optimizer can have a partitioned optimizer status. Note that optimizers must
                 consider gradient partitioning if applied; otherwise the result will be incorrect.
   2.2 (ZeRO-2): Use reduce instead of all-reduce in (1) to obtain only a partition of gradients.
"""
from mnm.ir import MNMSequential
from .optim import inline
from .. import distributed as dist
from .._ffi.pass_ import PartitionGradient, InferType
from ..model import Model, trace
from ..model.trace import _get_func_inputs


def with_data_parallel(model):
    """Enable data parallel and ZeRO to the model according to the configs in dist context."""

    class DataParallelWrapper(Model):
        """Data parallel model

        Parameters
        ----------
        model: Model
            The model with forward, backawrd, and partitioned output gradients.
        """

        def build(self, model):
            # pylint: disable=attribute-defined-outside-init, missing-function-docstring
            self.model = model

        @trace
        def forward(self, *args, **kwargs):
            # pylint: disable=protected-access, missing-function-docstring
            passes = []
            dctx = dist.get_context()
            # TODO: Refactor AutoDataParallel to let it work on the IR after InlineBackward
            # so that it can be applied here.
            # if dctx.enable_data_parallel:
            #     passes.append(AutoDataParallel())
            if dctx.zero_opt_level > 0:
                passes.append(InferType())
                passes.append(PartitionGradient(dctx.zero_opt_level, dctx.size, dctx.rank))

            record = self.model._internal(*args, **kwargs)
            mod = record.mod
            seq = MNMSequential(passes)
            mod = seq(mod)
            inputs = _get_func_inputs(record, args, kwargs)
            out = inline(mod["main"], inputs)
            y = out[0]
            dxs = out[1]
            return y, dxs

    return DataParallelWrapper(model)
