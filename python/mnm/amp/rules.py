# pylint: disable=unused-argument,missing-function-docstring
"""Pre-defined rules for different operators for automatic mixed precision."""
from enum import IntEnum

from mnm._lib import tvm, relay


class CastHintType(IntEnum):
    """Possible kinds of cast hint."""

    skip = 0
    float16 = 1
    float32 = 2


def register_op_cast_rule(op_name, cast_rule=None, level=10):
    """Register alter op layout function for an op

    Parameters
    ----------
    op_name : str
        The name of the operator

    cast_rule: function (args: List[Expr]) -> new_expr: List[int]
        The function for providing cast hint type

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FMNMCastRule", cast_rule, level)


@register_op_cast_rule("mnm.op.conv2d")
def op_cast_rule_conv2d(args):
    return [CastHintType.float16, CastHintType.float16] + [CastHintType.skip] * 4


# float16 ops
@register_op_cast_rule("mnm.op.avg_pool2d")
def op_cast_rule_avg_pool2d(args):
    return [CastHintType.float16] + [CastHintType.skip] * 6


@register_op_cast_rule("mnm.op.max_pool2d")
def op_cast_rule_max_pool2d(args):
    return [CastHintType.float16] + [CastHintType.skip] * 6


def op_cast_rule_matmul(args):
    return [CastHintType.float16, CastHintType.float16]


register_op_cast_rule("mnm.op.batch_matmul", op_cast_rule_matmul)
register_op_cast_rule("mnm.op.matmul", op_cast_rule_matmul)
register_op_cast_rule("mnm.op.matmul_nt", op_cast_rule_matmul)
register_op_cast_rule("mnm.op.dense", op_cast_rule_matmul)
register_op_cast_rule("mnm.op.matmul_tn", op_cast_rule_matmul)
register_op_cast_rule("mnm.op.matmul_tt", op_cast_rule_matmul)


# float32 ops
def op_cast_rule_batch_norm(args):
    return [CastHintType.float16] + [CastHintType.skip] * 6


register_op_cast_rule("mnm.op.batch_norm_train", op_cast_rule_batch_norm)
register_op_cast_rule("mnm.op.batch_norm_infer", op_cast_rule_batch_norm)


@register_op_cast_rule("mnm.op.log_softmax")
def op_cast_rule_log_softmax(args):
    return [CastHintType.float16, CastHintType.skip]


@register_op_cast_rule("mnm.op.nll_loss")
def op_cast_rule_nll_loss(args):
    return [CastHintType.float16, CastHintType.float16]


# widest ops
def op_cast_rule_widest_op(args):
    # assume Constant(s) appears after Var(s)
    ret = []
    count = 0
    use_f32 = False
    for arg in args:
        if isinstance(arg, relay.Constant):
            ret.append(CastHintType.skip)
        else:
            ttype = arg.checked_type
            if isinstance(ttype, tvm.ir.TensorType):
                count += 1
                if ttype.dtype == "float32":
                    use_f32 = True
            else:
                raise ValueError("Input argument is not a tensor.")
    ret = ([CastHintType.float32] * count if use_f32 else [CastHintType.float16] * count) + ret
    return ret


register_op_cast_rule("mnm.op.add", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.subtract", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.multiply", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.divide", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.mod", op_cast_rule_widest_op)
