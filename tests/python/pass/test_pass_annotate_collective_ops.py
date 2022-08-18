# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use,too-many-arguments
import pytest
import tvm

import raf
from raf.ir import PassContext
from raf._ffi.pass_ import AnnotateCollectiveOps
from raf.testing import with_dialect
from raf._core.ir_ext import extended_var
from raf.ir import ANFBuilder


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize(
    "comm_op,comm_args",
    [
        ["_allreduce", [raf.ir.const("sum")]],
        ["_reduce", [raf.ir.const(0), raf.ir.const("sum")]],
        ["_broadcast", [raf.ir.const(0)]],
    ],
)
@pytest.mark.parametrize("shape", [[128, 128], [64, 64], [32, 64, 128]])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_annotate_collective(comm_op, comm_args, shape, dtype):
    size = 1
    for axis in shape:
        size *= axis

    def construct_model_func():
        builder = ANFBuilder()
        x = extended_var("x", shape=shape, dtype=dtype)
        x_1 = builder.call(comm_op, [x] + comm_args)
        return tvm.relay.Function([x], builder.ret(x_1))

    mod = tvm.IRModule()
    mod["main"] = construct_model_func()
    mod = AnnotateCollectiveOps()(mod)

    assert tvm.ir.structural_equal(mod["main"], construct_model_func())


if __name__ == "__main__":
    pytest.main([__file__])
