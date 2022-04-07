# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long
"""Operator-specific type hints for automatic mixed precision.

The type hint function for each operator should return a list of type hints,
and the list size should be the same as the argument number.
Here are some type hint examples:

- PrimType("float32"): The argument must be in float32.
- PrimType("float16"): The argument should be in the specified AMP dtype (float16 in this case).
- PrimType(None): Do not change the dtype of this argument. It means if the argument has been
    casted to the AMP dtype, we need to cast it back.

The current list is based on TF:
github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h

Note that since the infer list is the majority, we make it as the default behavior
and do not need to specify them here.

TODO(@comaniac): We need to consider the accumulation dtype, which may be different to the output
dtype. However, we need to make sure the ops will use the desired dtype for accumulation in advance.
"""
# pylint: enable=line-too-long
# pylint: disable=unused-argument
from raf._lib import tvm
from tvm import relay
from tvm.ir import PrimType, TupleType


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
    return tvm.ir.register_op_attr(op_name, "FRAFCastRule", cast_rule, level)


def gen_hint_helper(etype, target_dtype):
    """A helper function to generate a type hint for the given type."""
    if isinstance(etype, tvm.ir.TensorType):
        return PrimType(target_dtype if "float" in etype.dtype else etype.dtype)
    if isinstance(etype, tvm.ir.TupleType):
        return TupleType([gen_hint_helper(field, target_dtype) for field in etype.fields])
    raise ValueError("Unsupported input type: %s" % str(etype))


def check_dtype(ttype, dtype):
    """Check whether the given type has the target dtype."""
    if isinstance(ttype, tvm.ir.TupleType):
        return all([check_dtype(field, dtype) for field in ttype.fields])
    assert isinstance(ttype, tvm.ir.TensorType)
    return ttype.dtype == dtype


def generic_cast(cast_to_amp, castable_arg_num_or_list):
    """The generic cast function that generates AMP type hints for inputs, and generates
    don't touch type hints for rest arguments.

    Parameters
    ----------
    cast_to_amp : bool
        Whether to cast all arguments to the AMP dtype.

    castable_arg_num_or_list : Union[int, List[int]]
        The first number or list of arguments that can be casted to the AMP dtype.

    Returns
    -------
    gen: Callable[[List[Expr], Type], List[Type]]
        The cast rule function.
    """
    if isinstance(castable_arg_num_or_list, int):
        castable_arg_list = range(castable_arg_num_or_list)
    else:
        assert isinstance(castable_arg_num_or_list, list), "Expected int or list, but got %s" % (
            type(castable_arg_num_or_list)
        )
        castable_arg_list = castable_arg_num_or_list

    def _gen(args, ret_type, amp_dtype):
        target_dtype = amp_dtype if cast_to_amp else None
        ret = []
        for idx, arg in enumerate(args):
            if idx in castable_arg_list:
                ret.append(gen_hint_helper(arg.checked_type, target_dtype))
            else:
                ret.append(PrimType(None))
        return ret

    return _gen


# Always cast.
register_op_cast_rule("raf.op.conv2d", generic_cast(True, 2))
register_op_cast_rule("raf.op.conv2d_dx", generic_cast(True, 3))
register_op_cast_rule("raf.op.conv2d_dw", generic_cast(True, 3))
register_op_cast_rule("raf.op.conv2d_transpose", generic_cast(True, 2))
register_op_cast_rule("raf.op.conv2d_transpose_dx", generic_cast(True, 3))
register_op_cast_rule("raf.op.conv2d_transpose_dw", generic_cast(True, 3))
register_op_cast_rule("raf.op.matmul", generic_cast(True, 2))
register_op_cast_rule("raf.op.dense", generic_cast(True, 2))
register_op_cast_rule("raf.op.matmul_nt", generic_cast(True, 2))
register_op_cast_rule("raf.op.matmul_tn", generic_cast(True, 2))
register_op_cast_rule("raf.op.matmul_tt", generic_cast(True, 2))
register_op_cast_rule("raf.op.batch_matmul", generic_cast(True, 2))
register_op_cast_rule("raf.op.batch_matmul_nt", generic_cast(True, 2))
register_op_cast_rule("raf.op.batch_matmul_tn", generic_cast(True, 2))
register_op_cast_rule("raf.op.batch_matmul_tt", generic_cast(True, 2))

# Never cast.
register_op_cast_rule("raf.op.arange", generic_cast(False, 3))
register_op_cast_rule("raf.op.exp", generic_cast(False, 1))
register_op_cast_rule("raf.op.power", generic_cast(False, 1))
register_op_cast_rule("raf.op.softmax", generic_cast(False, 1))
register_op_cast_rule("raf.op.softmax_dx", generic_cast(False, 2))
register_op_cast_rule("raf.op.lans", generic_cast(False, 2))
register_op_cast_rule("raf.op.log_softmax", generic_cast(False, 1))
register_op_cast_rule("raf.op.log_softmax_dx", generic_cast(False, 2))
register_op_cast_rule("raf.op.erf", generic_cast(False, 1))
register_op_cast_rule("raf.op.erf_dx", generic_cast(False, 3))
register_op_cast_rule("raf.op.gelu", generic_cast(False, 1))
register_op_cast_rule("raf.op.gelu_dx", generic_cast(False, 3))
register_op_cast_rule("raf.op.smooth_l1_loss", generic_cast(False, 2))
register_op_cast_rule("raf.op.smooth_l1_loss_dpred", generic_cast(False, 2))
register_op_cast_rule("raf.op.smooth_l1_loss_dtrue", generic_cast(False, 2))
register_op_cast_rule("raf.op.nll_loss", generic_cast(False, 2))
register_op_cast_rule("raf.op.nll_loss_dpred", generic_cast(False, 3))
register_op_cast_rule("raf.op.nll_loss_dtrue", generic_cast(False, 3))
register_op_cast_rule("raf.op.cross_entropy", generic_cast(False, 2))
register_op_cast_rule("raf.op.cross_entropy_dpred", generic_cast(False, 2))
register_op_cast_rule("raf.op.cross_entropy_dtrue", generic_cast(False, 2))

# embedding_dx/take_dx has accuracy issue and its performance does not improve significantly
# over float32, so never cast.
register_op_cast_rule("raf.op.take_dx", generic_cast(3, False))
register_op_cast_rule("raf.op.embedding_dx", generic_cast(2, False))

# FIXME: These ops should support float16, but the current TVM code results in
# either runtime error or mismatch outputs.
register_op_cast_rule("raf.op.atan", generic_cast(False, 1))
register_op_cast_rule("raf.op.tanh", generic_cast(False, 1))
register_op_cast_rule("raf.op.tanh_dx", generic_cast(False, 3))
register_op_cast_rule("raf.op.rsqrt", generic_cast(False, 1))

# These ops needs to accumulate the result in float32, so we never cast them,
# and expect they will be fused with the cast ops.
register_op_cast_rule("raf.op.multiply", generic_cast(False, 2))
register_op_cast_rule("raf.op.sum", generic_cast(False, 1))
register_op_cast_rule("raf.op.gather_dx", generic_cast(False, 4))
register_op_cast_rule("raf.op.gather_nd", generic_cast(False, 1))
register_op_cast_rule("raf.op.gather_nd_dx", generic_cast(False, 3))


def infer_cast(castable_arg_num_or_list):
    """The cast rule is inferred by the dtype of current arguments and output:
    1. If the original output dtype is not float32 (e.g., int32/bool/tuple), then do not touch.
    2. If more than a half args are casted to the AMP dtype, then cast all args to the AMP dtype.
    3. Otherwise keep all arguments untouched.

    Parameters
    ----------
    castable_arg_num_or_list : Union[int, List[int]]
        The first number or list of arguments that can be casted to the AMP dtype.

    Returns
    -------
    gen: Callable[[List[Expr], Type], List[Type]]
        The cast rule function.
    """
    if isinstance(castable_arg_num_or_list, int):
        castable_arg_list = range(castable_arg_num_or_list)
    else:
        assert isinstance(castable_arg_num_or_list, list), "Expected int or list, but got %s" % (
            type(castable_arg_num_or_list)
        )
        castable_arg_list = castable_arg_num_or_list

    def _gen(args, ret_type, amp_dtype):
        if not castable_arg_list or isinstance(ret_type, tvm.ir.TupleType):
            # Not castable or just follow the current input dtype.
            target_dtype = None
        else:
            n_amp = 0
            n_fp32 = 0
            for idx in castable_arg_list:
                n_amp += 1 if check_dtype(args[idx].checked_type, amp_dtype) else 0
                n_fp32 += 1 if check_dtype(args[idx].checked_type, "float32") else 0
            target_dtype = "float32" if n_fp32 > n_amp else amp_dtype

        ret = []
        for idx, arg in enumerate(args):
            if idx in castable_arg_list:
                ret.append(gen_hint_helper(arg.checked_type, target_dtype))
            else:
                ret.append(PrimType(None))
        return ret

    return _gen


# Infer cast.
register_op_cast_rule("raf.op.max_pool2d", infer_cast(1))
register_op_cast_rule("raf.op.avg_pool2d", infer_cast(1))
register_op_cast_rule("raf.op.adaptive_max_pool2d", infer_cast(1))
register_op_cast_rule("raf.op.adaptive_avg_pool2d", infer_cast(1))
register_op_cast_rule("raf.op.pad", infer_cast(1))
register_op_cast_rule("raf.op.max_pool2d_dx", infer_cast(3))
register_op_cast_rule("raf.op.avg_pool2d_dx", infer_cast(3))
register_op_cast_rule("raf.op.adaptive_max_pool2d_dx", infer_cast(3))
register_op_cast_rule("raf.op.adaptive_avg_pool2d_dx", infer_cast(3))
register_op_cast_rule("raf.op.batch_flatten", infer_cast(1))
register_op_cast_rule("raf.op.negative", infer_cast(1))
register_op_cast_rule("raf.op.logical_not", infer_cast(1))
register_op_cast_rule("raf.op.relu", infer_cast(1))
register_op_cast_rule("raf.op.copy", infer_cast(1))
register_op_cast_rule("raf.op.abs", infer_cast(1))
register_op_cast_rule("raf.op.all", infer_cast(1))
register_op_cast_rule("raf.op.any", infer_cast(1))
register_op_cast_rule("raf.op.ceil", infer_cast(1))
register_op_cast_rule("raf.op.cos", infer_cast(1))
register_op_cast_rule("raf.op.sin", infer_cast(1))
register_op_cast_rule("raf.op.sign", infer_cast(1))
register_op_cast_rule("raf.op.round", infer_cast(1))
register_op_cast_rule("raf.op.floor", infer_cast(1))
register_op_cast_rule("raf.op.log", infer_cast(1))
register_op_cast_rule("raf.op.log2", infer_cast(1))
register_op_cast_rule("raf.op.sigmoid", infer_cast(1))
register_op_cast_rule("raf.op.sqrt", infer_cast(1))
register_op_cast_rule("raf.op.relu_dx", infer_cast(3))
register_op_cast_rule("raf.op.sigmoid_dx", infer_cast(3))
register_op_cast_rule("raf.op.sqrt_dx", infer_cast(3))
register_op_cast_rule("raf.op.floor_divide", infer_cast(2))
register_op_cast_rule("raf.op.mod", infer_cast(2))
register_op_cast_rule("raf.op.less", infer_cast(2))
register_op_cast_rule("raf.op.greater", infer_cast(2))
register_op_cast_rule("raf.op.less_equal", infer_cast(2))
register_op_cast_rule("raf.op.greater_equal", infer_cast(2))
register_op_cast_rule("raf.op.equal", infer_cast(2))
register_op_cast_rule("raf.op.not_equal", infer_cast(2))
register_op_cast_rule("raf.op.maximum", infer_cast(2))
register_op_cast_rule("raf.op.minimum", infer_cast(2))
register_op_cast_rule("raf.op.right_shift", infer_cast(2))
register_op_cast_rule("raf.op.left_shift", infer_cast(2))
register_op_cast_rule("raf.op.trunc", infer_cast(1))
register_op_cast_rule("raf.op.mesh_grid", infer_cast(2))
register_op_cast_rule("raf.op.reshape", infer_cast(1))
register_op_cast_rule("raf.op.reshape_like", infer_cast(1))
register_op_cast_rule("raf.op.resize2d", infer_cast(1))
register_op_cast_rule("raf.op.ndarray_size", infer_cast(1))
register_op_cast_rule("raf.op.transpose", infer_cast(1))
register_op_cast_rule("raf.op.transpose_dx", infer_cast(1))
register_op_cast_rule("raf.op.collapse_sum_like", infer_cast(1))
register_op_cast_rule("raf.op.sum_dx", infer_cast(2))
register_op_cast_rule("raf.op.argmax", infer_cast(1))
register_op_cast_rule("raf.op.argmin", infer_cast(1))
register_op_cast_rule("raf.op.prod", infer_cast(1))
register_op_cast_rule("raf.op.prod_dx", infer_cast(2))
register_op_cast_rule("raf.op.max", infer_cast(1))
register_op_cast_rule("raf.op.min", infer_cast(1))
register_op_cast_rule("raf.op.mean", infer_cast(1))
register_op_cast_rule("raf.op.mean_dx", infer_cast(1))
register_op_cast_rule("raf.op.get_reduce_axis", infer_cast(2))
register_op_cast_rule("raf.op.get_kept_dims", infer_cast(2))
register_op_cast_rule("raf.op.sgd", infer_cast(1))
register_op_cast_rule("raf.op.shape", infer_cast(1))
register_op_cast_rule("raf.op.swap_axis", infer_cast(1))
register_op_cast_rule("raf.op.repeat", infer_cast(1))
register_op_cast_rule("raf.op.repeat_dx", infer_cast(3))
register_op_cast_rule("raf.op.expand_dims", infer_cast(1))
register_op_cast_rule("raf.op.threefry_generate", infer_cast(1))
register_op_cast_rule("raf.op.threefry_split", infer_cast(1))
register_op_cast_rule("raf.op.strided_slice", infer_cast(1))
register_op_cast_rule("raf.op.strided_slice_dx", infer_cast(1))
register_op_cast_rule("raf.op.sequence_mask", infer_cast(1))
register_op_cast_rule("raf.op.reverse_sequence", infer_cast(1))
register_op_cast_rule("raf.op.reverse", infer_cast(1))
register_op_cast_rule("raf.op.broadcast_to", infer_cast(1))
register_op_cast_rule("raf.op.broadcast_to_like", infer_cast(1))
register_op_cast_rule("raf.op.squeeze", infer_cast(1))
register_op_cast_rule("raf.op.stack", infer_cast(1))
register_op_cast_rule("raf.op.scatter", infer_cast(1))
register_op_cast_rule("raf.op.scatter_dx", infer_cast(3))
register_op_cast_rule("raf.op.clip", infer_cast(1))
register_op_cast_rule("raf.op.clip_dx", infer_cast(2))
register_op_cast_rule("raf.op.get_valid_counts", infer_cast(1))
register_op_cast_rule("raf.op.bias_add", infer_cast(2))
register_op_cast_rule("raf.op._contrib_dropout", infer_cast(1))
register_op_cast_rule("raf.op._contrib_dropout_dx", infer_cast(1))
register_op_cast_rule("raf.op.non_max_suppression", infer_cast(1))
register_op_cast_rule("raf.op._allreduce", infer_cast(1))
register_op_cast_rule("raf.op._allgather", infer_cast(1))
register_op_cast_rule("raf.op._reduce_scatter", infer_cast(1))
register_op_cast_rule("raf.op.argsort", infer_cast(1))
register_op_cast_rule("raf.op.sort", infer_cast(1))
register_op_cast_rule("raf.op.full", infer_cast(0))
register_op_cast_rule("raf.op.full_like", infer_cast(1))
register_op_cast_rule("raf.op.where", infer_cast(3))
register_op_cast_rule("raf.op.logical_and", infer_cast(2))
register_op_cast_rule("raf.op.topk", infer_cast(1))
register_op_cast_rule("raf.op.zeros", infer_cast(0))
register_op_cast_rule("raf.op.zeros_like", infer_cast(1))
register_op_cast_rule("raf.op.ones", infer_cast(0))
register_op_cast_rule("raf.op.ones_like", infer_cast(1))
register_op_cast_rule("raf.op.one_hot", infer_cast(0))
register_op_cast_rule("raf.op.argwhere", infer_cast(1))
register_op_cast_rule("raf.op.upper_bound.argwhere", infer_cast(1))
register_op_cast_rule("raf.op.roi_align", infer_cast(2))
register_op_cast_rule("raf.op.roi_align_dx", infer_cast(2))
register_op_cast_rule("raf.op.gather", infer_cast(1))
register_op_cast_rule("raf.op.divide", infer_cast(2))
register_op_cast_rule("raf.op.cumsum", infer_cast(1))
register_op_cast_rule("raf.op.size", infer_cast(1))
register_op_cast_rule("raf.op.numel", infer_cast(1))
register_op_cast_rule("raf.op.shape_as_tensor", infer_cast(1))
register_op_cast_rule("raf.op.embedding", infer_cast(2))
register_op_cast_rule("raf.op.take", infer_cast(2))

# Special cases.


def op_cast_cast(args, ret_type, amp_dtype):
    """For cast ops, we only need to put the correct return dtype to the type hint
    so that another cast op will be generated if it does not match to the requirement of
    the next op.
    """
    return [PrimType(None), PrimType(None)]


register_op_cast_rule("raf.op.cast", op_cast_cast)
register_op_cast_rule("raf.op.cast_like", op_cast_cast)


def op_cast_binary_ufunc(args, ret_type, amp_dtype):
    """The 3rd and 4th arguments of binary ufunc scheme are out and where, which may be
    nullptr that cannot be casted. On the other hand, if they are not nullptr (constant node),
    then we need to treat them as the first two arguments.
    """
    assert isinstance(
        ret_type, tvm.ir.TensorType
    ), "Op with binary_ufunc schema should not return a tuple"
    ret_dtype = ret_type.dtype
    cast_to_amp = ret_dtype == "float32"
    if cast_to_amp:
        cast_to_amp = any([check_dtype(arg.checked_type, amp_dtype) for arg in args[:2]])

    # This op inplace updates an existing tensor, so the type hints must align to it.
    if not isinstance(args[2], relay.Constant):
        assert isinstance(args[2].checked_type, tvm.ir.TensorType)
        cast_to_amp = args[2].checked_type.dtype == amp_dtype
        ret_dtype = args[2].checked_type.dtype

    target_dtype = amp_dtype if cast_to_amp else None

    ret = []
    for arg in args[:2]:
        ret.append(gen_hint_helper(arg.checked_type, target_dtype))

    # out: same as the return type.
    ret.append(PrimType(None) if isinstance(args[2], relay.Constant) else PrimType(ret_dtype))
    ret.append(PrimType(None))  # where: do not touch
    return ret


register_op_cast_rule("raf.op.add", op_cast_binary_ufunc)
register_op_cast_rule("raf.op.subtract", op_cast_binary_ufunc)


def op_cast_adv_index(args, ret_type, amp_dtype):
    """adv_index/adv_index_dx are the only ops that need to take a tuple with different dtypes
    as an input.
    """
    return [TupleType([PrimType(amp_dtype), PrimType(None), PrimType(None)])]


register_op_cast_rule("raf.op.adv_index", op_cast_adv_index)


def op_cast_adv_index_dx(args, ret_type, amp_dtype):
    """adv_index/adv_index_dx are the only ops that need to take a tuple with different dtypes
    as an input.
    """
    return [PrimType(amp_dtype), TupleType([PrimType(amp_dtype), PrimType(None), PrimType(None)])]


register_op_cast_rule("raf.op.adv_index_dx", op_cast_adv_index_dx)


def op_cast_norm(data_num):
    """Scale/bias tensors of normalization layers have to be in float32."""

    def _gen_rules(args, ret_type, amp_dtype):
        ret = [PrimType(amp_dtype) for _ in range(data_num)]
        ret += [PrimType(None) for _ in range(len(args) - data_num)]
        return ret

    return _gen_rules


register_op_cast_rule("raf.op.batch_norm_infer", op_cast_norm(1))
register_op_cast_rule("raf.op.batch_norm_train", op_cast_norm(1))

# TODO(@comaniac): batch_norm_train_dxwb produces different results as PyTorch BatchNorm backward
# and we have not figured out the reason. However, it does not affect the convergence of AMP models
# so we still cast it.
register_op_cast_rule("raf.op.batch_norm_train_dxwb", op_cast_norm(2))


register_op_cast_rule("raf.op.layer_norm", infer_cast(1))
register_op_cast_rule("raf.op.layer_norm_dx", infer_cast(3))


def op_cast_layer_norm_train(args, ret_type, amp_dtype):
    """Always follow the dtype for the 1st arg because the latency of taking FP16 and FP32
    are basically the same."""
    return [
        PrimType(args[0].checked_type.dtype),
        PrimType("float32"),
        PrimType("float32"),
        PrimType(None),
        PrimType(None),
    ]


def op_cast_layer_norm_train_dx(args, ret_type, amp_dtype):
    """It has args in order (x, scale, dy, mean, invvar, axis, eps)."""
    ret = [
        PrimType(args[0].checked_type.dtype),
        PrimType("float32"),
        PrimType(None),
        PrimType("float32"),
    ]
    ret += [PrimType(None) for _ in range(len(args) - 4)]
    return ret


register_op_cast_rule("raf.op.layer_norm_train", op_cast_layer_norm_train)
register_op_cast_rule("raf.op.layer_norm_train_dx", op_cast_layer_norm_train_dx)


def op_cast_concatenate(args, ret_type, amp_dtype):
    """Concatenate may have too many inputs that exceeds the GPU register when using the injective
    schedule with float16, so we make a heuristic that prevents concat from being executed with
    float16 if it has too many inputs.
    """
    in_types = args[0].checked_type.fields
    cast_to_amp = sum([check_dtype(t, amp_dtype) for t in in_types]) > len(in_types) // 2
    cast_to_amp &= len(in_types) <= 5
    target_dtype = amp_dtype if cast_to_amp else "float32"

    ret = []
    ret += [TupleType([PrimType(target_dtype) for _ in in_types])]  # Input tuple.
    ret += [PrimType(None)]  # axis.
    return ret


register_op_cast_rule("raf.op.concatenate", op_cast_concatenate)
register_op_cast_rule("raf.op.concatenate_dx", op_cast_concatenate)


def op_cast_split(args, ret_type, amp_dtype):
    """Split generates a tuple output but its behavior is quite simple, so it is safe
    to always let it follow the argument dtype.
    """
    target_dtype = amp_dtype if check_dtype(args[0].checked_type, amp_dtype) else None

    ret = [gen_hint_helper(args[0].checked_type, target_dtype)]
    ret += [PrimType(None) for _ in range(len(args) - 1)]
    return ret


register_op_cast_rule("raf.op.split", op_cast_split)
