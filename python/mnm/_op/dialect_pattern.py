"""Define dialect fusion patterns."""
from .dialect import register_pattern
from .._lib import is_op, wildcard, has_dtype

MATMUL_OPS = [
    "mnm.op.dense",
    "mnm.op.matmul",
    "mnm.op.matmul_nt",
    "mnm.op.matmul_tn",
    "mnm.op.matmul_tt"
]

BATCH_MATMUL_OPS = [
    "mnm.op.batch_matmul",
    "mnm.op.batch_matmul_nt",
    "mnm.op.batch_matmul_tn",
    "mnm.op.batch_matmul_tt"
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
    ret = []
    for _ in range(n):
        ret.append(wildcard())
    return ret


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
    scaled_bias = is_op("mnm.op.multiply")(*n_wildcards(2))
    bias = scaled_bias | wildcard()
    # pattern: matmul+scaled_bias or matmul+bias
    with_bias = is_op("mnm.op.add")(matmul, bias, *n_wildcards(2))
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


def _cutlass_conv2d_fusion(dtype=None):
    act_ops = ["mnm.op.relu"]
    conv = _call_conv2d(dtype)
    # pattern: conv2d+bias
    with_bias = is_op("mnm.op.add")(conv, *n_wildcards(3))
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
