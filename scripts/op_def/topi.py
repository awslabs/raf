# pylint: disable=missing-class-docstring,missing-function-docstring
"""Operator definition for TOPI (TVM Operator Inventory)."""
import sys

__all__ = ["OP_MAP"]

# pylint: disable=line-too-long

OP_MAP = {
    "mnm.op.tvm.abs": ["abs", "", "kElemWise"],
    "mnm.op.tvm.add": ["add", "", "kBroadcast"],
    "mnm.op.tvm.all": ["all", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.any": ["any", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.arange": ["arange", "relay.attrs.ArangeAttrs", "kOpaque"],
    "mnm.op.tvm.argmax": ["argmax", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.argmin": ["argmin", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.argsort": ["argsort", "relay.attrs.ArgsortAttrs", "kOpaque"],
    "mnm.op.tvm.atan": ["atan", "", "kElemWise"],
    "mnm.op.tvm.broadcast_to": ["broadcast_to", "", "kBroadcast"],
    "mnm.op.tvm.broadcast_to_like": ["broadcast_to_like", "", "kBroadcast"],
    "mnm.op.tvm.cast": ["cast", "relay.attrs.CastAttrs", "kElemWise"],
    "mnm.op.tvm.cast_like": ["cast_like", "", "kElemWise"],
    "mnm.op.tvm.ceil": ["ceil", "", "kElemWise"],
    "mnm.op.tvm.clip": ["clip", "relay.attrs.ClipAttrs", "kElemWise"],
    "mnm.op.tvm.collapse_sum_like": ["collapse_sum_like", "", "kCommReduce"],
    "mnm.op.tvm.concatenate": ["concatenate", "relay.attrs.ConcatenateAttrs", "kInjective"],
    "mnm.op.tvm.copy": ["copy", "", "kElemWise"],
    "mnm.op.tvm.cos": ["cos", "", "kElemWise"],
    "mnm.op.tvm.divide": ["divide", "", "kBroadcast"],
    "mnm.op.tvm.equal": ["equal", "", "kBroadcast"],
    "mnm.op.tvm.erf": ["erf", "", "kElemWise"],
    "mnm.op.tvm.exp": ["exp", "", "kElemWise"],
    "mnm.op.tvm.expand_dims": ["expand_dims", "relay.attrs.ExpandDimsAttrs", "kBroadcast"],
    "mnm.op.tvm.floor": ["floor", "", "kElemWise"],
    "mnm.op.tvm.floor_divide": ["floor_divide", "", "kBroadcast"],
    "mnm.op.tvm.floor_mod": ["floor_mod", "", "kBroadcast"],
    "mnm.op.tvm.gather": ["gather", "", "kInjective"],
    "mnm.op.tvm.gather_nd": ["gather_nd", "", "kInjective"],
    "mnm.op.tvm.greater": ["greater", "", "kBroadcast"],
    "mnm.op.tvm.greater_equal": ["greater_equal", "", "kBroadcast"],
    "mnm.op.tvm.layout_transform": ["layout_transform", "relay.attrs.LayoutTransformAttrs", "kInjective"],
    "mnm.op.tvm.left_shift": ["left_shift", "", "kBroadcast"],
    "mnm.op.tvm.less": ["less", "", "kBroadcast"],
    "mnm.op.tvm.less_equal": ["less_equal", "", "kBroadcast"],
    "mnm.op.tvm.log": ["log", "", "kElemWise"],
    "mnm.op.tvm.log2": ["log2", "", "kElemWise"],
    "mnm.op.tvm.log_softmax": ["nn.log_softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "mnm.op.tvm.logical_and": ["logical_and", "", "kBroadcast"],
    "mnm.op.tvm.logical_not": ["logical_not", "", "kElemWise"],
    "mnm.op.tvm.logical_or": ["logical_or", "", "kBroadcast"],
    "mnm.op.tvm.max": ["max", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.maximum": ["maximum", "", "kBroadcast"],
    "mnm.op.tvm.mean": ["mean", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.min": ["min", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.minimum": ["minimum", "", "kBroadcast"],
    "mnm.op.tvm.mod": ["mod", "", "kBroadcast"],
    "mnm.op.tvm.multiply": ["multiply", "", "kBroadcast"],
    "mnm.op.tvm.negative": ["negative", "", "kElemWise"],
    "mnm.op.tvm.bias_add": ["nn.bias_add", "relay.attrs.BiasAddAttrs", "kBroadcast"],
    "mnm.op.tvm.not_equal": ["not_equal", "", "kBroadcast"],
    "mnm.op.tvm.one_hot": ["one_hot", "relay.attrs.OneHotAttrs", "kOutEWiseFusable"],
    "mnm.op.tvm.ones": ["ones", "relay.attrs.InitOpAttrs", "kElemWise"],
    "mnm.op.tvm.ones_like": ["ones_like", "", "kElemWise"],
    "mnm.op.tvm.power": ["power", "", "kBroadcast"],
    "mnm.op.tvm.prod": ["prod", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.reinterpret": ["reinterpret", "relay.attrs.CastAttrs", "kElemWise"],
    "mnm.op.tvm.repeat": ["repeat", "relay.attrs.RepeatAttrs", "kBroadcast"],
    "mnm.op.tvm.reverse": ["reverse", "relay.attrs.ReverseAttrs", "kInjective"],
    "mnm.op.tvm.right_shift": ["right_shift", "", "kBroadcast"],
    "mnm.op.tvm.round": ["round", "", "kElemWise"],
    "mnm.op.tvm.rsqrt": ["rsqrt", "", "kElemWise"],
    "mnm.op.tvm.reshape": ["reshape", "relay.attrs.ReshapeAttrs", "kInjective"],
    "mnm.op.tvm.sequence_mask": ["sequence_mask", "relay.attrs.SequenceMaskAttrs", "kInjective"],
    "mnm.op.tvm.reverse_sequence": ["reverse_sequence", "relay.attrs.ReverseSequenceAttrs", "kInjective"],
    "mnm.op.tvm.scatter": ["scatter", "relay.attrs.ScstterAttrs", "kOpaque"],
    "mnm.op.tvm.sigmoid": ["sigmoid", "", "kElemWise"],
    "mnm.op.tvm.sign": ["sign", "", "kElemWise"],
    "mnm.op.tvm.sin": ["sin", "", "kElemWise"],
    "mnm.op.tvm.slice_like": ["slice_like", "relay.attrs.SliceLikeAttrs", "kInjective"],
    "mnm.op.tvm.split": ["split", "relay.attrs.SplitAttrs", "kInjective"],
    "mnm.op.tvm.sqrt": ["sqrt", "", "kElemWise"],
    "mnm.op.tvm.squeeze": ["squeeze", "relay.attrs.SqueezeAttrs", "kInjective"],
    "mnm.op.tvm.stack": ["stack", "relay.attrs.StackAttrs", "kInjective"],
    "mnm.op.tvm.strided_slice": ["strided_slice", "relay.attrs.StridedSliceAttrs", "kInjective"],
    "mnm.op.tvm.subtract": ["subtract", "", "kBroadcast"],
    "mnm.op.tvm.take": ["take", "relay.attrs.TakeAttrs", "kInjective"],
    "mnm.op.tvm.tanh": ["tanh", "", "kElemWise"],
    "mnm.op.tvm.tile": ["tile", "relay.attrs.TileAttrs", "kBroadcast"],
    "mnm.op.tvm.topk": ["topk", "relay.attrs.TopkAttrs", "kOpaque"],
    "mnm.op.tvm.transpose": ["transpose", "relay.attrs.TransposeAttrs", "kInjective"],
    "mnm.op.tvm.trunc": ["trunc", "", "kElemWise"],
    "mnm.op.tvm.threefry_generate": ["random.threefry_generate", "relay.attrs.ThreefryGenerateAttrs", "kOpaque"],
    "mnm.op.tvm.threefry_split": ["random.threefry_split", "", "kOpaque"],
    "mnm.op.tvm.variance": ["variance", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "mnm.op.tvm.where": ["where", "", "kBroadcast"],
    "mnm.op.tvm.zeros": ["zeros", "relay.attrs.InitOpAttrs", "kElemWise"],
    "mnm.op.tvm.zeros_like": ["zeros_like", "", "kElemWise"],
    "mnm.op.tvm.batch_matmul_nt": ["nn.batch_matmul", "", "kOpaque"],
    "mnm.op.tvm.dense": ["nn.dense", "", "kOpaque"],
    "mnm.op.tvm.softmax": ["nn.softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "mnm.op.tvm.relu": ["nn.relu", "", "kInjective"],
    "mnm.op.tvm.avg_pool2d": ["nn.avg_pool2d", "", "kOpaque"],
    "mnm.op.tvm.avg_pool2d_dx": ["nn.avg_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.tvm.max_pool2d": ["nn.max_pool2d", "", "kOpaque"],
    "mnm.op.tvm.max_pool2d_dx": ["nn.max_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.tvm.adaptive_avg_pool2d": ["nn.adaptive_avg_pool2d", "", "kOpaque"],
    "mnm.op.tvm.adaptive_avg_pool2d_dx": ["nn.avg_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.tvm.adaptive_max_pool2d": ["nn.adaptive_max_pool2d", "", "kOpaque"],
    "mnm.op.tvm.adaptive_max_pool2d_dx": ["nn.max_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "mnm.op.tvm.get_valid_counts": ["get_valid_counts", "", "kInjective"],
    "mnm.op.tvm.non_max_suppression": ["non_max_suppression", "relay.attrs.NonMaxSuppressionAttrs", "kInjective"],
    "mnm.op.tvm.batch_flatten": ["nn.batch_flatten", "", "kInjective"],
    "mnm.op.tvm.device_copy": ["device_copy", "relay.attrs.DeviceCopyAttrs", "kOpaque"],
    "mnm.op.tvm.roi_align": ["vision.roi_align", "relay.attrs.ROIAlignAttrs", "kOutEWiseFusable"],
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
    "nn.bias_add": "mnm.op.tvm.bias_add",
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
        mnm_op_name = MNM_OP_NAME.get(op_name, "mnm.op.tvm." + op_name)
        print(f'    "{mnm_op_name}": ["{op_name}", "{attrs}", "{pattern}"],')
    print("}\n")
    print("# pylint: enable=line-too-long\n")
