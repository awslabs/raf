"""Operator-specific type hints for automatic mixed precision."""
from mnm._lib import tvm, relay
from tvm.ir import PrimType, TupleType, TypeVar


def register_op_cast_rule(op_name, cast_rule=None, level=10):
    """Register a custom casting rule function for an op.

    Parameters
    ----------
    op_name : str
        The name of the operator

    cast_rule: Callable[[List[Expr], Type], List[Type]]
        The function for providing cast hint type. The last hint type is the return type.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FMNMCastRule", cast_rule, level)


def op_cast_fp32(args, ret_type):
    """The rule for ops that can only be executed with float32."""
    def _gen(etype):
        if isinstance(etype, tvm.ir.TensorType):
            return PrimType("float32")
        if isinstance(etype, tvm.ir.TupleType):
            return TupleType([_gen(field) for field in etype.fields])
        raise ValueError("Unsupported input type: %s" % str(etype))

    ret = [
        _gen(arg.checked_type) if not isinstance(arg, relay.Constant) else PrimType(None)
        for arg in args
    ]
    ret += [_gen(ret_type)]
    return ret


register_op_cast_rule("mnm.op.erf", op_cast_fp32)


def op_cast_batch_norm(data_num, out_num):
    """Scale/bias tensor have to be in float32."""

    def _gen_rules(args, _):
        ret = [TypeVar("amp") for _ in range(data_num)]
        ret += [PrimType(None) for _ in range(len(args) - data_num)]
        if out_num == 1:
            ret.append(TypeVar("amp"))
        else:
            ret += [TupleType([TypeVar("amp"), PrimType("float32"), PrimType("float32")])]
        return ret

    return _gen_rules


register_op_cast_rule("mnm.op.batch_norm_infer", op_cast_batch_norm(1, 1))
register_op_cast_rule("mnm.op.batch_norm_train", op_cast_batch_norm(1, 3))
register_op_cast_rule("mnm.op.batch_norm_train_dxwb", op_cast_batch_norm(2, 3))


def op_cast_rule_widest_op(args, _):
    """This rule guarantees that 2 input and 1 output tensors are in the same dtype."""
    ret = []

    # Determine whether to use float32 or the AMP type based on the arguments.
    use_f32 = True
    for arg in args:
        if not isinstance(arg, relay.Constant):
            ttype = arg.checked_type
            if isinstance(ttype, tvm.ir.TensorType):
                if ttype.dtype != "float32":
                    use_f32 = False
                    break
            else:
                raise ValueError("Input argument is not a tensor.")

    for arg in args[:2]:
        ret.append(PrimType("float32") if use_f32 else TypeVar("amp"))
    for arg in args[2:]:
        if isinstance(arg, relay.Constant):
            ret.append(PrimType(None))
        else:
            ret.append(PrimType("float32") if use_f32 else TypeVar("amp"))
    ret.append(PrimType("float32") if use_f32 else TypeVar("amp"))
    return ret


register_op_cast_rule("mnm.op.add", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.subtract", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.multiply", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.divide", op_cast_rule_widest_op)
register_op_cast_rule("mnm.op.mod", op_cast_rule_widest_op)
