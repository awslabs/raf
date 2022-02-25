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
@pytest.mark.parametrize("use_memory_copy_ops", [True, False])
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
@pytest.mark.parametrize("n", [2, 3, 4])
def test_annotate_collective(use_memory_copy_ops, comm_op, comm_args, shape, dtype, n):
    size = 1
    for axis in shape:
        size *= axis
    sizes = [size] * n
    shapes = shape * n
    shape_indices = list(range(len(shape), n * len(shape) + 1, len(shape)))

    def construct_model_func():
        builder = ANFBuilder()
        x = extended_var("x", shape=shape, dtype=dtype)
        x_1 = builder.make_tuple([x] * n)
        x_2 = builder.call(comm_op, [x_1] + comm_args)
        x_3 = builder.get_tuple_item(x_2, 0)
        return tvm.relay.Function([x], builder.ret(x_3))

    def expected():
        builder = ANFBuilder()
        x = extended_var("x", shape=shape, dtype=dtype)
        x_1 = builder.make_tuple([x] * n)
        x_1 = builder.call("fuse_tensor", [x_1])
        x_1 = builder.make_tuple([x_1])
        x_2 = builder.call(comm_op, [x_1] + comm_args)
        x_2 = builder.call(
            "defuse_tensor",
            [x_2, raf.ir.const(sizes), raf.ir.const(shapes), raf.ir.const(shape_indices)],
        )
        x_3 = builder.get_tuple_item(x_2, 0)
        return tvm.relay.Function([x], builder.ret(x_3))

    config = {"raf.annotate_collective_ops.use_memory_copy_ops": use_memory_copy_ops}
    with PassContext(config=config):
        mod = tvm.IRModule()
        mod["main"] = construct_model_func()
        mod = AnnotateCollectiveOps()(mod)

    if use_memory_copy_ops:
        assert tvm.ir.structural_equal(mod["main"], expected())
    else:
        assert tvm.ir.structural_equal(mod["main"], construct_model_func())


if __name__ == "__main__":
    pytest.main([__file__])
