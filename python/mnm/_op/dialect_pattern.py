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

"""Define dialect fusion patterns."""
from .dialect import register_pattern
from ..ir.dataflow_pattern import is_op, wildcard, is_constant, has_dtype, has_shape
from .._core.value import StringValue

MATMUL_OPS = [
    "mnm.op.dense",
    "mnm.op.matmul",
    "mnm.op.matmul_nt",
    "mnm.op.matmul_tn",
    "mnm.op.matmul_tt",
]

BATCH_MATMUL_OPS = [
    "mnm.op.batch_matmul",
    "mnm.op.batch_matmul_nt",
    "mnm.op.batch_matmul_tn",
    "mnm.op.batch_matmul_tt",
]


def is_ops(ops):
    """Create a pattern that match to a list of ops."""
    if isinstance(ops, str):
        return is_op(ops)
    assert isinstance(ops, (tuple, list)), "ops must be string/list/tuple"
    op_pat = None
    for op in ops:
        if op_pat is None:
            op_pat = is_op(op)
        else:
            op_pat = op_pat | is_op(op)
    return op_pat


def n_wildcards(n):
    """Create a list of wildcard patterns."""
    return [wildcard() for _ in range(n)]


def n_null_constant(n):
    """Create a list of wildcard patterns."""
    return [is_constant(None) for _ in range(n)]


def call_binary_ops(ops, dtype=None):
    """Create a call pattern to the given binary ops."""
    if dtype is None:
        x, y = wildcard(), wildcard()
    else:
        assert isinstance(dtype, str)
        x, y = has_dtype(dtype), has_dtype(dtype)
    return is_ops(ops)(x, y)


def _cutlass_matmul_fusion(matmul_ops, dtype=None):
    act_ops = ["mnm.op.relu", "mnm.op.gelu"]
    # matmul
    matmul = call_binary_ops(matmul_ops, dtype)
    # bias
    bias = wildcard()
    beta = has_shape(()) | has_shape((1,))
    scaled_bias = is_op("mnm.op.multiply")(beta, bias)
    bias = scaled_bias | bias
    # pattern: matmul+scaled_bias or matmul+bias
    with_bias = is_op("mnm.op.add")(matmul, bias, *n_null_constant(2))
    # pattern: matmul+(scaled_)bias+act or matmul+act
    with_act = is_ops(act_ops)(with_bias | matmul)
    # We exclude the single matmul op pattern as ther perf of cutlass is worse than cublas
    return with_act | with_bias


def _call_conv2d(dtype=None):
    if dtype is None:
        x, w = wildcard(), wildcard()
    else:
        assert isinstance(dtype, str)
        x, w = has_dtype(dtype), has_dtype(dtype)
    return is_op("mnm.op.conv2d")(x, w, *n_wildcards(7))


def _call_conv2d_dxw(dtype=None):
    ops = ["mnm.op.conv2d_dx", "mnm.op.conv2d_dw"]
    if dtype is None:
        x_or_w, y, dy = wildcard(), wildcard(), wildcard()
    else:
        assert isinstance(dtype, str)
        x_or_w, y, dy = has_dtype(dtype), has_dtype(dtype), has_dtype(dtype)
    return is_ops(ops)(x_or_w, y, dy, *n_wildcards(5))


def _cutlass_conv2d_fusion():
    act_ops = ["mnm.op.relu"]
    conv = is_op("mnm.op.conv2d")(
        *n_wildcards(6),
        is_constant(StringValue("NHWC")),
        is_constant(StringValue("OHWI")),
        is_constant(StringValue("NHWC"))
    )
    # pattern: conv2d+bias
    with_bias = is_op("mnm.op.add")(conv, wildcard(), *n_null_constant(2))
    # pattern: conv2d+bias+act || conv2d+act
    with_act = is_ops(act_ops)(with_bias | conv)
    return with_act | with_bias


def _call_pool2d_dx():
    pool_ops = ["mnm.op.max_pool2d_dx", "mnm.op.avg_pool2d_dx"]
    return is_ops(pool_ops)(*n_wildcards(9))


# pool2d_dx
register_pattern(_call_pool2d_dx(), "cudnn", 50, "pool2d_dx")

# conv2d_dx, conv2d_dw
register_pattern(_call_conv2d_dxw(), "cudnn", 40, "conv2d_dxw")

# conv2d
register_pattern(_cutlass_conv2d_fusion(), "cutlass", 30, "conv2d_fusion")
register_pattern(_call_conv2d(), "cudnn", 29, "conv2d")

# batch_matmul
register_pattern(_cutlass_matmul_fusion(BATCH_MATMUL_OPS), "cutlass", 20, "batch_matmul_fusion")
register_pattern(call_binary_ops(BATCH_MATMUL_OPS), "cublas", 19, "batch_matmul")
register_pattern(call_binary_ops(BATCH_MATMUL_OPS), "cutlass", 18, "batch_matmul")

# matmul / dense
register_pattern(_cutlass_matmul_fusion(MATMUL_OPS), "cutlass", 10, "matmul_fusion")
register_pattern(call_binary_ops(MATMUL_OPS), "cublas", 9, "matmul")
register_pattern(call_binary_ops(MATMUL_OPS), "cutlass", 8, "matmul")
