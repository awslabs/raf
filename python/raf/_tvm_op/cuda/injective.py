# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals, invalid-name, no-member
"""Schedule for some injective operators, based on 3rdparty/tvm/python/tvm/topi/cuda/reduction.py"""
import tvm
from tvm import te
import tvm.topi.utils as utils


def schedule_injective_from_existing(sch, out):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """
    fused = sch[out].fuse(*sch[out].op.axis)
    num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
    # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES occurs if max_block is 256 when running test case:
    # test_conv2d("cuda", "float32", ((4, 256, 32, 32), (64, 256, 1, 1)), 1, 1, 0)
    max_block = 255

    # vectorize on fp16 data type. This allows to better utilize the memory
    # bandwidth.
    vector_width = 4 if out.dtype == "float16" else 1

    is_dynamic_output = False
    for dim in out.shape:
        if not isinstance(dim, tvm.tir.IntImm):
            is_dynamic_output = True
            break

    out_len = utils.prod(out.shape)

    try:
        const_size = utils.get_const_int(out_len)
        need_block_split = const_size > max_block * num_thread * vector_width
    except ValueError:
        # const_size is not constant integer.
        need_block_split = False
        const_size = 0

    if vector_width > 1:
        fused, v = sch[out].split(fused, vector_width)
        sch[out].vectorize(v)

    if need_block_split:
        xo, xi = sch[out].split(fused, factor=num_thread * max_block)
        bx, tx = sch[out].split(xi, factor=num_thread)

        sch[out].reorder(bx, tx, xo)
        sch[out].bind(bx, te.thread_axis("blockIdx.x"))
        sch[out].bind(tx, te.thread_axis("threadIdx.x"))
    else:
        # Use less threads for dynamic shape ops to avoid runtime error.
        if is_dynamic_output:
            num_thread //= 2
        if const_size != 0 and const_size < num_thread:
            bx, tx = sch[out].split(fused, factor=const_size)
        else:
            bx, tx = sch[out].split(fused, factor=num_thread)
        sch[out].bind(tx, te.thread_axis("threadIdx.x"))
        sch[out].bind(bx, te.thread_axis("blockIdx.x"))

    return sch


def schedule_injective(outs):
    """Schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    tvm.te.schedule.AutoInlineInjective(s)
    for out in outs:
        if not utils.is_empty_shape(out.shape):
            schedule_injective_from_existing(s, out)
    return s


schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
