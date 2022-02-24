# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring
"""Operator definition for TOPI (TVM Operator Inventory)."""
import sys

__all__ = ["OP_MAP"]

# pylint: disable=line-too-long

OP_MAP = {
    "raf.op.tvm.abs": ["abs", "", "kElemWise"],
    "raf.op.tvm.add": ["add", "", "kBroadcast"],
    "raf.op.tvm.all": ["all", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.any": ["any", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.arange": ["arange", "relay.attrs.ArangeAttrs", "kOpaque"],
    "raf.op.tvm.argmax": ["argmax", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.argmin": ["argmin", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.argsort": ["argsort", "relay.attrs.ArgsortAttrs", "kOpaque"],
    "raf.op.tvm.atan": ["atan", "", "kElemWise"],
    "raf.op.tvm.broadcast_to": ["broadcast_to", "", "kBroadcast"],
    "raf.op.tvm.broadcast_to_like": ["broadcast_to_like", "", "kBroadcast"],
    "raf.op.tvm.cast": ["cast", "relay.attrs.CastAttrs", "kElemWise"],
    "raf.op.tvm.cast_like": ["cast_like", "", "kElemWise"],
    "raf.op.tvm.ceil": ["ceil", "", "kElemWise"],
    "raf.op.tvm.clip": ["clip", "relay.attrs.ClipAttrs", "kElemWise"],
    "raf.op.tvm.collapse_sum_like": ["collapse_sum_like", "", "kCommReduce"],
    "raf.op.tvm.cumsum": ["cumsum", "relay.attrs.ScanopAttrs", "kOpaque"],
    "raf.op.tvm.concatenate": ["concatenate", "relay.attrs.ConcatenateAttrs", "kInjective"],
    "raf.op.tvm.copy": ["copy", "", "kElemWise"],
    "raf.op.tvm.cos": ["cos", "", "kElemWise"],
    "raf.op.tvm.divide": ["divide", "", "kBroadcast"],
    "raf.op.tvm.equal": ["equal", "", "kBroadcast"],
    "raf.op.tvm.erf": ["erf", "", "kElemWise"],
    "raf.op.tvm.exp": ["exp", "", "kElemWise"],
    "raf.op.tvm.expand_dims": ["expand_dims", "relay.attrs.ExpandDimsAttrs", "kBroadcast"],
    "raf.op.tvm.floor": ["floor", "", "kElemWise"],
    "raf.op.tvm.floor_divide": ["floor_divide", "", "kBroadcast"],
    "raf.op.tvm.floor_mod": ["floor_mod", "", "kBroadcast"],
    "raf.op.tvm.gather": ["gather", "", "kInjective"],
    "raf.op.tvm.gather_nd": ["gather_nd", "", "kInjective"],
    "raf.op.tvm.greater": ["greater", "", "kBroadcast"],
    "raf.op.tvm.greater_equal": ["greater_equal", "", "kBroadcast"],
    "raf.op.tvm.layout_transform": [
        "layout_transform",
        "relay.attrs.LayoutTransformAttrs",
        "kInjective",
    ],
    "raf.op.tvm.left_shift": ["left_shift", "", "kBroadcast"],
    "raf.op.tvm.less": ["less", "", "kBroadcast"],
    "raf.op.tvm.less_equal": ["less_equal", "", "kBroadcast"],
    "raf.op.tvm.log": ["log", "", "kElemWise"],
    "raf.op.tvm.log2": ["log2", "", "kElemWise"],
    "raf.op.tvm.log_softmax": ["nn.log_softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "raf.op.tvm.logical_and": ["logical_and", "", "kBroadcast"],
    "raf.op.tvm.logical_not": ["logical_not", "", "kElemWise"],
    "raf.op.tvm.logical_or": ["logical_or", "", "kBroadcast"],
    "raf.op.tvm.max": ["max", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.maximum": ["maximum", "", "kBroadcast"],
    "raf.op.tvm.mean": ["mean", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.min": ["min", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.minimum": ["minimum", "", "kBroadcast"],
    "raf.op.tvm.mod": ["mod", "", "kBroadcast"],
    "raf.op.tvm.multiply": ["multiply", "", "kBroadcast"],
    "raf.op.tvm.negative": ["negative", "", "kElemWise"],
    "raf.op.tvm.bias_add": ["nn.bias_add", "relay.attrs.BiasAddAttrs", "kBroadcast"],
    "raf.op.tvm.not_equal": ["not_equal", "", "kBroadcast"],
    "raf.op.tvm.one_hot": ["one_hot", "relay.attrs.OneHotAttrs", "kOutEWiseFusable"],
    "raf.op.tvm.ones": ["ones", "relay.attrs.InitOpAttrs", "kElemWise"],
    "raf.op.tvm.ones_like": ["ones_like", "", "kElemWise"],
    "raf.op.tvm.power": ["power", "", "kBroadcast"],
    "raf.op.tvm.prod": ["prod", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.reinterpret": ["reinterpret", "relay.attrs.CastAttrs", "kElemWise"],
    "raf.op.tvm.repeat": ["repeat", "relay.attrs.RepeatAttrs", "kBroadcast"],
    "raf.op.tvm.reverse": ["reverse", "relay.attrs.ReverseAttrs", "kInjective"],
    "raf.op.tvm.right_shift": ["right_shift", "", "kBroadcast"],
    "raf.op.tvm.round": ["round", "", "kElemWise"],
    "raf.op.tvm.rsqrt": ["rsqrt", "", "kElemWise"],
    "raf.op.tvm.reshape": ["reshape", "relay.attrs.ReshapeAttrs", "kInjective"],
    "raf.op.tvm.sequence_mask": ["sequence_mask", "relay.attrs.SequenceMaskAttrs", "kInjective"],
    "raf.op.tvm.reverse_sequence": [
        "reverse_sequence",
        "relay.attrs.ReverseSequenceAttrs",
        "kInjective",
    ],
    "raf.op.tvm.scatter": ["scatter", "relay.attrs.ScstterAttrs", "kOpaque"],
    "raf.op.tvm.sigmoid": ["sigmoid", "", "kElemWise"],
    "raf.op.tvm.sign": ["sign", "", "kElemWise"],
    "raf.op.tvm.sin": ["sin", "", "kElemWise"],
    "raf.op.tvm.slice_like": ["slice_like", "relay.attrs.SliceLikeAttrs", "kInjective"],
    "raf.op.tvm.split": ["split", "relay.attrs.SplitAttrs", "kInjective"],
    "raf.op.tvm.sqrt": ["sqrt", "", "kElemWise"],
    "raf.op.tvm.squeeze": ["squeeze", "relay.attrs.SqueezeAttrs", "kInjective"],
    "raf.op.tvm.stack": ["stack", "relay.attrs.StackAttrs", "kInjective"],
    "raf.op.tvm.strided_slice": ["strided_slice", "relay.attrs.StridedSliceAttrs", "kInjective"],
    "raf.op.tvm.subtract": ["subtract", "", "kBroadcast"],
    "raf.op.tvm.take": ["take", "relay.attrs.TakeAttrs", "kInjective"],
    "raf.op.tvm.tanh": ["tanh", "", "kElemWise"],
    "raf.op.tvm.tile": ["tile", "relay.attrs.TileAttrs", "kBroadcast"],
    "raf.op.tvm.topk": ["topk", "relay.attrs.TopkAttrs", "kOpaque"],
    "raf.op.tvm.transpose": ["transpose", "relay.attrs.TransposeAttrs", "kInjective"],
    "raf.op.tvm.trunc": ["trunc", "", "kElemWise"],
    "raf.op.tvm.threefry_generate": [
        "random.threefry_generate",
        "relay.attrs.ThreefryGenerateAttrs",
        "kOpaque",
    ],
    "raf.op.tvm.threefry_split": ["random.threefry_split", "", "kOpaque"],
    "raf.op.tvm.variance": ["variance", "relay.attrs.ReduceAttrs", "kCommReduce"],
    "raf.op.tvm.where": ["where", "", "kBroadcast"],
    "raf.op.tvm.zeros": ["zeros", "relay.attrs.InitOpAttrs", "kElemWise"],
    "raf.op.tvm.zeros_like": ["zeros_like", "", "kElemWise"],
    "raf.op.tvm.batch_matmul_nt": ["nn.batch_matmul", "", "kOpaque"],
    "raf.op.tvm.dense": ["nn.dense", "", "kOpaque"],
    "raf.op.tvm.softmax": ["nn.softmax", "relay.attrs.SoftmaxAttrs", "kOpaque"],
    "raf.op.tvm.relu": ["nn.relu", "", "kInjective"],
    "raf.op.tvm.avg_pool2d": ["nn.avg_pool2d", "", "kOpaque"],
    "raf.op.tvm.avg_pool2d_dx": ["nn.avg_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "raf.op.tvm.max_pool2d": ["nn.max_pool2d", "", "kOpaque"],
    "raf.op.tvm.max_pool2d_dx": ["nn.max_pool2d_grad", "relay.attrs.MaxPool2DAttrs", "kOpaque"],
    "raf.op.tvm.adaptive_avg_pool2d": ["nn.adaptive_avg_pool2d", "", "kOpaque"],
    "raf.op.tvm.adaptive_avg_pool2d_dx": [
        "nn.avg_pool2d_grad",
        "relay.attrs.MaxPool2DAttrs",
        "kOpaque",
    ],
    "raf.op.tvm.adaptive_max_pool2d": ["nn.adaptive_max_pool2d", "", "kOpaque"],
    "raf.op.tvm.adaptive_max_pool2d_dx": [
        "nn.max_pool2d_grad",
        "relay.attrs.MaxPool2DAttrs",
        "kOpaque",
    ],
    "raf.op.tvm.get_valid_counts": ["get_valid_counts", "", "kInjective"],
    "raf.op.tvm.non_max_suppression": [
        "non_max_suppression",
        "relay.attrs.NonMaxSuppressionAttrs",
        "kInjective",
    ],
    "raf.op.tvm.batch_flatten": ["nn.batch_flatten", "", "kInjective"],
    "raf.op.tvm.device_copy": ["device_copy", "relay.attrs.DeviceCopyAttrs", "kOpaque"],
    "raf.op.tvm.roi_align": ["vision.roi_align", "relay.attrs.ROIAlignAttrs", "kOutEWiseFusable"],
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

RAF_OP_NAME = {
    "nn.bias_add": "raf.op.tvm.bias_add",
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
        if op_name.startswith("raf."):
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
            print("[Skip]", op_name, ":", ", ".join(skip_reasons), file=sys.stderr)
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
        raf_op_name = RAF_OP_NAME.get(op_name, "raf.op.tvm." + op_name)
        print(f'    "{raf_op_name}": ["{op_name}", "{attrs}", "{pattern}"],')
    print("}\n")
    print("# pylint: enable=line-too-long\n")
