# pylint: disable=missing-class-docstring,missing-function-docstring
"""Operator definition for TOPI (TVM Operator Inventory)."""
import sys

__all__ = ["OP_MAP"]

# pylint: disable=line-too-long

OP_MAP = {
    "mnm.op.abs": ["abs", "", "kElemWise"],
    "mnm.op.add": ["add", "", "kBroadcast"],
    "mnm.op.all": ["all", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.any": ["any", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.arange": ["arange", "relay.attrs.ArangeAttrs", "kOpaque"],
    "mnm.op.argmax": ["argmax", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.argmin": ["argmin", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.argsort": ["argsort", "relay.attrs.ArgsortAttrs", "kOpaque"],
    "mnm.op.argwhere": ["argwhere", "relay.attrs.ArgWhereAttrs", "kOpaque"],
    "mnm.op.atan": ["atan", "", "kElemWise"],
    "mnm.op.broadcast_to": ["broadcast_to", "", "kBroadcast"],
    "mnm.op.broadcast_to_like": ["broadcast_to_like", "", "kBroadcast"],
    "mnm.op.cast": ["cast", "relay.attrs.CastAttrs", "kElemWise"],
    "mnm.op.cast_like": ["cast_like", "", "kElemWise"],
    "mnm.op.ceil": ["ceil", "", "kElemWise"],
    "mnm.op.clip": ["clip", "relay.attrs.ClipAttrs", "kElemWise"],
    "mnm.op.collapse_sum_like": ["collapse_sum_like", "", "kCommReduce"],
    "mnm.op.concatenate": ["concatenate", "relay.attrs.ConcatenateAttrs", "kInjective"],
    "mnm.op.copy": ["copy", "", "kElemWise"],
    "mnm.op.cos": ["cos", "", "kElemWise"],
    "mnm.op.divide": ["divide", "", "kBroadcast"],
    "mnm.op.equal": ["equal", "", "kBroadcast"],
    "mnm.op.erf": ["erf", "", "kElemWise"],
    "mnm.op.exp": ["exp", "", "kElemWise"],
    "mnm.op.expand_dims": ["expand_dims", "relay.attrs.ExpandDimsAttrs", "kBroadcast"],
    "mnm.op.floor": ["floor", "", "kElemWise"],
    "mnm.op.floor_divide": ["floor_divide", "", "kBroadcast"],
    "mnm.op.floor_mod": ["floor_mod", "", "kBroadcast"],
    "mnm.op.full": ["full", "relay.attrs.InitOpAttrs", "kElemWise"],
    "mnm.op.full_like": ["full_like", "", "kElemWise"],
    "mnm.op.gather": ["gather", "", "kInjective"],
    "mnm.op.gather_nd": ["gather_nd", "", "kInjective"],
    "mnm.op.greater": ["greater", "", "kBroadcast"],
    "mnm.op.greater_equal": ["greater_equal", "", "kBroadcast"],
    "mnm.op.image.resize": ["image.resize", "relay.attrs.ResizeAttrs", "kInjective"],
    "mnm.op.layout_transform": ["layout_transform", "relay.attrs.LayoutTransformAttrs", "kInjective"],
    "mnm.op.left_shift": ["left_shift", "", "kBroadcast"],
    "mnm.op.less": ["less", "", "kBroadcast"],
    "mnm.op.less_equal": ["less_equal", "", "kBroadcast"],
    "mnm.op.log": ["log", "", "kElemWise"],
    "mnm.op.log_softmax": ["nn.log_softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "mnm.op.logical_and": ["logical_and", "", "kBroadcast"],
    "mnm.op.logical_not": ["logical_not", "", "kElemWise"],
    "mnm.op.logical_or": ["logical_or", "", "kBroadcast"],
    "mnm.op.max": ["max", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.maximum": ["maximum", "", "kBroadcast"],
    "mnm.op.mean": ["mean", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.min": ["min", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.minimum": ["minimum", "", "kBroadcast"],
    "mnm.op.mod": ["mod", "", "kBroadcast"],
    "mnm.op.multiply": ["multiply", "", "kBroadcast"],
    "mnm.op.negative": ["negative", "", "kElemWise"],
    "mnm.op.bias_add": ["nn.bias_add", "relay.attrs.BiasAddAttrs", "kBroadcast"],
    "mnm.op.not_equal": ["not_equal", "", "kBroadcast"],
    "mnm.op.one_hot": ["one_hot", "relay.attrs.OneHotAttrs", "kOutEWiseFusable"],
    "mnm.op.ones": ["ones", "relay.attrs.InitOpAttrs", "kElemWise"],
    "mnm.op.ones_like": ["ones_like", "", "kElemWise"],
    "mnm.op.power": ["power", "", "kBroadcast"],
    "mnm.op.prod": ["prod", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.reinterpret": ["reinterpret", "relay.attrs.CastAttrs", "kElemWise"],
    "mnm.op.repeat": ["repeat", "relay.attrs.RepeatAttrs", "kBroadcast"],
    "mnm.op.reverse": ["reverse", "relay.attrs.ReverseAttrs", "kInjective"],
    "mnm.op.right_shift": ["right_shift", "", "kBroadcast"],
    "mnm.op.round": ["round", "", "kElemWise"],
    "mnm.op.rsqrt": ["rsqrt", "", "kElemWise"],
    "mnm.op.reshape": ["reshape", "relay.attrs.ReshapeAttrs", "kInjective"],
    "mnm.op.sequence_mask": ["sequence_mask", "relay.attrs.SequenceMaskAttrs", "kInjective"],
    "mnm.op.reverse_sequence": ["reverse_sequence", "relay.attrs.ReverseSequenceAttrs", "kInjective"],
    "mnm.op.sigmoid": ["sigmoid", "", "kElemWise"],
    "mnm.op.sign": ["sign", "", "kElemWise"],
    "mnm.op.sin": ["sin", "", "kElemWise"],
    "mnm.op.slice_like": ["slice_like", "relay.attrs.SliceLikeAttrs", "kInjective"],
    "mnm.op.split": ["split", "relay.attrs.SplitAttrs", "kInjective"],
    "mnm.op.sqrt": ["sqrt", "", "kElemWise"],
    "mnm.op.squeeze": ["squeeze", "relay.attrs.SqueezeAttrs", "kInjective"],
    "mnm.op.stack": ["stack", "relay.attrs.StackAttrs", "kInjective"],
    "mnm.op.strided_slice": ["strided_slice", "relay.attrs.StridedSliceAttrs", "kInjective"],
    "mnm.op.subtract": ["subtract", "", "kBroadcast"],
    "mnm.op.take": ["take", "relay.attrs.TakeAttrs", "kInjective"],
    "mnm.op.tanh": ["tanh", "", "kElemWise"],
    "mnm.op.tile": ["tile", "relay.attrs.TileAttrs", "kBroadcast"],
    "mnm.op.topk": ["topk", "relay.attrs.TopkAttrs", "kOpaque"],
    "mnm.op.transpose": ["transpose", "relay.attrs.TransposeAttrs", "kInjective"],
    "mnm.op.trunc": ["trunc", "", "kElemWise"],
    "mnm.op.variance": ["variance", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.where": ["where", "", "kBroadcast"],
    "mnm.op.zeros": ["zeros", "relay.attrs.InitOpAttrs", "kElemWise"],
    "mnm.op.zeros_like": ["zeros_like", "", "kElemWise"],
    "mnm.op.batch_matmul": ["nn.batch_matmul", "", "kOpaque"],
    "mnm.op.dense": ["nn.dense", "", "kOpaque"],
    "mnm.op.softmax": ["nn.softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "mnm.op.pad": ["nn.pad", "relay.attrs.PadAttrs", "kOpaque"],
    "mnm.op.relu": ["nn.relu", "", "kInjective"],
    "mnm.op.avg_pool2d": ["nn.avg_pool2d", "", "kOpaque"],
    "mnm.op.avg_pool2d_dx": ["nn.avg_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.max_pool2d": ["nn.max_pool2d", "", "kOpaque"],
    "mnm.op.max_pool2d_dx": ["nn.max_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.get_valid_counts": ["get_valid_counts", "", "kInjective"],
    "mnm.op.non_max_suppression": ["non_max_suppression", "relay.attrs.NonMaxSuppressionAttrs", "kInjective"],
    "mnm.op.compiler_begin": ["annotation.compiler_begin", "relay.attrs.CompilerAttrs", "kOpaque"],
    "mnm.op.compiler_end": ["annotation.compiler_end", "relay.attrs.CompilerAttrs", "kOpaque"],
    "mnm.op.batch_flatten": ["nn.batch_flatten", "", "kInjective"],
    "mnm.op.device_copy": ["device_copy", "relay.attrs.DeviceCopyAttrs", "kOpaque"],
}

# pylint: enable=line-too-long

PREFIX_BLACK_LIST = {
    "annotation.",
    "memory.",
    "qnn.",
    "vision.",
    "nn.",
    "contrib.",
    "_contrib_reverse_reshape",
}

BLACK_LIST = {
    "on_device",
    "device_copy",
    "relay.op.annotation.simulated_quantize",
}

WHILTE_LIST = {
    "nn.bias_add",
}

MNM_OP_NAME = {
    "nn.bias_add": "mnm.op.bias_add",
}


def collect_op():
    # pylint: disable=import-outside-toplevel
    import tvm
    from tvm import relay

    pattern_map = {
        0: "kElemWise",
        1: "kBroadcast",
        2: "kInjective",
        3: "kCommReduce",
        4: "kOutEWiseFusable",
        7: "kTuple",
        8: "kOpaque",
    }
    list_op = tvm.get_global_func("relay.op._ListOpNames")
    get_op = tvm.get_global_func("relay.op._GetOp")

    def is_black_listed(op_name):
        if op_name.startswith("mnm."):
            return True
        if op_name in WHILTE_LIST:
            assert op_name not in BLACK_LIST
            return False
        if op_name in BLACK_LIST:
            print("[Skip]", op_name, ": Blacklisted", file=sys.stderr)
            return True
        for prefix in PREFIX_BLACK_LIST:
            if op_name.startswith(prefix):
                print("[Skip]", op_name, ": Blacklisted", file=sys.stderr)
                return True
        return False

    result = []
    for op_name in list_op():
        op_name = op_name.value
        if is_black_listed(op_name):
            continue
        op: relay.Op = get_op(op_name)  # pylint: disable=no-member
        assert op.name == op_name
        attrs = op.attrs_type_key
        fcompute = op.get_attr("FTVMCompute")
        fschedule = op.get_attr("FTVMSchedule")
        pattern = op.get_attr("TOpPattern")
        skip_reasons = []
        if not fcompute:
            skip_reasons.append("No-FTVMCompute")
        if not fschedule:
            skip_reasons.append("No-FTVMSchedule")
        if pattern is None:
            skip_reasons.append("No-TOpPattern")
        if skip_reasons:
            print("[Skip]",
                  op_name,
                  ":",
                  ", ".join(skip_reasons),
                  file=sys.stderr)
            continue
        if not attrs:
            attrs = ""
        pattern = pattern_map[pattern]
        result.append((op_name, attrs, pattern))
    return result


def main():
    ops = collect_op()
    print("# pylint: disable=line-too-long\n")
    print("OP_MAP = {")
    for op_name, attrs, pattern in ops:
        mnm_op_name = MNM_OP_NAME.get(op_name, "mnm.op." + op_name)
        print(f'    "{mnm_op_name}": ["{op_name}", "{attrs}", "{pattern}"],')
    print("}\n")
    print("# pylint: enable=line-too-long\n")
