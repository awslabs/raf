/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/transfrom.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "tvm/relay/attrs/transform.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY("arange", "raf.op.arange",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ArangeAttrs>();
                    raf_args.push_back(
                        MakeConstant(StringValue::make(DLDataType2String(relay_attrs->dtype))));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("adv_index", "raf.op.adv_index",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    return args;
                  });

RAF_OP_FROM_RELAY("repeat", "raf.op.repeat",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<RepeatAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->repeats.IntValue())));
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("take", "raf.op.take",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<TakeAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->mode)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("sequence_mask", "raf.op.sequence_mask",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<SequenceMaskAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->mask_value)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("reverse", "raf.op.reverse",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ReverseAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("reverse_sequence", "raf.op.reverse_sequence",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ReverseSequenceAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->seq_axis.IntValue())));
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->batch_axis.IntValue())));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("broadcast_to", "raf.op.broadcast_to",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<InitOpAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value())));
                    return raf_args;
                  });

RAF_GENERIC_ATTR_OP_FROM_RELAY("broadcast_to_like", "raf.op.broadcast_to_like");

RAF_GENERIC_ATTR_OP_FROM_RELAY("collapse_sum_like", "raf.op.collapse_sum_like");

RAF_GENERIC_ATTR_OP_FROM_RELAY("reshape_like", "raf.op.reshape_like");

RAF_OP_FROM_RELAY("transpose", "raf.op.transpose",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<TransposeAttrs>();
                    if (relay_attrs->axes.defined()) {
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axes)));
                    } else {
                      Array<Integer> empty;
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(empty)));
                    }
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("split", "raf.op.split",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<SplitAttrs>();
                    auto indices_or_sections = relay_attrs->indices_or_sections;
                    if (const auto* scalar = indices_or_sections.as<IntImmNode>()) {
                      auto val = scalar->value;
                      raf_args.push_back(MakeConstant(ScalarValue::make(val)));
                    } else if (const auto* arr = indices_or_sections.as<ArrayNode>()) {
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(*arr)));
                    } else {
                      CHECK(false) << "Fail to convert split: Unknown indices_or_sections type";
                    }
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("concatenate", "raf.op.concatenate",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ConcatenateAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("stack", "raf.op.stack",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<StackAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("clip", "raf.op.clip",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ClipAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->a_min)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->a_max)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("cast", "raf.op.cast",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<CastAttrs>();
                    raf_args.push_back(MakeConstant(
                        StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
                    return raf_args;
                  });

RAF_GENERIC_ATTR_OP_FROM_RELAY("cast_like", "raf.op.cast_like");

RAF_OP_FROM_RELAY("gather", "raf.op.gather",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    const auto* relay_attrs = attrs.as<GatherAttrs>();

                    // Relay args are (data, indices) and RAF args are (data, axis, indices).
                    raf_args.push_back(args[0]);
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    raf_args.push_back(args[1]);
                    return raf_args;
                  });

RAF_GENERIC_ATTR_OP_FROM_RELAY("gather_nd", "raf.op.gather_nd");

RAF_OP_FROM_RELAY("squeeze", "raf.op.squeeze",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<SqueezeAttrs>();
                    if (relay_attrs->axis.defined()) {
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
                    } else {
                      Array<Integer> empty;
                      raf_args.push_back(MakeConstant(ArrayToIntTuple(empty)));
                    }
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("reshape", "raf.op.reshape",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ReshapeAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->newshape)));
                    raf_args.push_back(MakeConstant(BoolValue::make(false)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("expand_dims", "raf.op.expand_dims",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ExpandDimsAttrs>();
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->num_newaxis)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("ones", "raf.op.ones",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    const auto* relay_attrs = attrs.as<InitOpAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value())));
                    raf_args.push_back(MakeConstant(
                        StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
                    raf_args.push_back(MakeConstant(StringValue::make("cpu")));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("full", "raf.op.full",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    const auto* konst = GetKonstFromValueMap(args[0], val_map);
                    CHECK(konst) << "'fill_value' must be a const tensor.";
                    raf_args.push_back(MakeConstant(Constant2ScalarValue<double>(konst)));

                    const auto* relay_attrs = attrs.as<InitOpAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value())));
                    raf_args.push_back(MakeConstant(
                        StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("full_like", "raf.op.full_like",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args;
                    raf_args.push_back(args[0]);
                    const auto* konst = GetKonstFromValueMap(args[1], val_map);
                    CHECK(konst) << "'fill_value' must be a const tensor.";
                    raf_args.push_back(MakeConstant(Constant2ScalarValue<double>(konst)));
                    return raf_args;
                  });

RAF_OP_FROM_RELAY("strided_slice", "raf.op.strided_slice",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<StridedSliceAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->begin.value())));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->end.value())));
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides.value())));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->slice_mode)));
                    return raf_args;
                  });

RAF_GENERIC_ATTR_OP_FROM_RELAY("where", "raf.op.where");

RAF_GENERIC_ATTR_OP_FROM_RELAY("argwhere", "raf.op.argwhere");

RAF_OP_FROM_RELAY("cumsum", "raf.op.cumsum",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ScanopAttrs>();
                    raf_args.push_back(
                        MakeConstant(ScalarValue::make(relay_attrs->axis.IntValue())));
                    raf_args.push_back(MakeConstant(
                        StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));

                    bool exclusive = false;
                    if (relay_attrs->exclusive.defined()) {
                      exclusive = bool(relay_attrs->exclusive);
                    }
                    raf_args.push_back(MakeConstant(BoolValue::make(exclusive)));
                    return raf_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace raf
