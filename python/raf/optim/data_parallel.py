# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
from raf.ir import RAFSequential
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
            dcfg = dist.get_config()
            comm = dist.get_communicator()
            record = self.model._internal(*args, **kwargs)
            mod = record.mod

            # TODO: Refactor AutoDataParallel to let it work on the IR after InlineBackward
            # so that it can be applied here.
            # if dcfg.enable_data_parallel:
            #     passes.append(AutoDataParallel())
            if dcfg.zero_opt_level > 0:
                passes = []
                passes.append(InferType())
                passes.append(
                    PartitionGradient(
                        dcfg.zero_opt_level, comm.size, comm.rank, dcfg.group_bucket_size
                    )
                )
                seq = RAFSequential(passes, name="with_data_parallel")
                mod = seq(mod)
            inputs = _get_func_inputs(record, args, kwargs)
            out = inline(mod["main"], inputs)
            y = out[0]
            dxs = out[1]
            return y, dxs

    return DataParallelWrapper(model)
