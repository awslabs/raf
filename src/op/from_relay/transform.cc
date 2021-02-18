/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/transfrom.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/transform.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("repeat", "mnm.op.repeat", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<RepeatAttrs>();
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->repeats)));
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("take", "mnm.op.take", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<TakeAttrs>();
  CHECK_EQ(relay_attrs->mode, "clip") << "Failed to convert take: Only support clip mode";
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("sequence_mask", "mnm.op.sequence_mask",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<SequenceMaskAttrs>();
                    mnm_args.push_back(MakeConstant(FloatValue::make(relay_attrs->mask_value)));
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("reverse", "mnm.op.reverse", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<ReverseAttrs>();
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("reverse_sequence", "mnm.op.reverse_sequence",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ReverseSequenceAttrs>();
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->seq_axis)));
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->batch_axis)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("broadcast_to", "mnm.op.broadcast_to",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<InitOpAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value())));
                    return mnm_args;
                  });

MNM_GENERIC_ATTR_OP_FROM_RELAY("broadcast_to_like", "mnm.op.broadcast_to_like");

MNM_OP_FROM_RELAY("transpose", "mnm.op.transpose",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<TransposeAttrs>();
                    if (relay_attrs->axes.defined()) {
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axes)));
                    } else {
                      Array<Integer> empty;
                      mnm_args.push_back(MakeConstant(ArrayToIntTuple(empty)));
                    }
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("split", "mnm.op.split", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<SplitAttrs>();
  auto indices_or_sections = relay_attrs->indices_or_sections;
  if (const auto* scalar = indices_or_sections.as<IntImmNode>()) {
    auto val = scalar->value;
    mnm_args.push_back(MakeConstant(IntValue::make(val)));
  } else if (const auto* arr = indices_or_sections.as<ArrayNode>()) {
    mnm_args.push_back(MakeConstant(ArrayToIntTuple(*arr)));
  } else {
    CHECK(false) << "Fail to convert split: Unknown indices_or_sections type";
  }
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("concatenate", "mnm.op.concatenate",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ConcatenateAttrs>();
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("stack", "mnm.op.stack", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<StackAttrs>();
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("clip", "mnm.op.clip", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<ClipAttrs>();
  mnm_args.push_back(MakeConstant(FloatValue::make(relay_attrs->a_min)));
  mnm_args.push_back(MakeConstant(FloatValue::make(relay_attrs->a_max)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("cast", "mnm.op.cast", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<CastAttrs>();
  mnm_args.push_back(
      MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
  return mnm_args;
});

MNM_GENERIC_ATTR_OP_FROM_RELAY("cast_like", "mnm.op.cast_like");

MNM_OP_FROM_RELAY("gather", "mnm.op.gather", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args;
  const auto* relay_attrs = attrs.as<GatherAttrs>();

  // Relay args are (data, indices) and Meta args are (data, axis, indices).
  mnm_args.push_back(args[0]);
  mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
  mnm_args.push_back(args[1]);
  return mnm_args;
});

MNM_GENERIC_ATTR_OP_FROM_RELAY("gather_nd", "mnm.op.gather_nd");

MNM_OP_FROM_RELAY("squeeze", "mnm.op.squeeze", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<SqueezeAttrs>();
  mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->axis)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("reshape", "mnm.op.reshape", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<ReshapeAttrs>();
  mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->newshape)));
  mnm_args.push_back(MakeConstant(BoolValue::make(false)));
  return mnm_args;
});

MNM_OP_FROM_RELAY("expand_dims", "mnm.op.expand_dims",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ExpandDimsAttrs>();
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->axis)));
                    mnm_args.push_back(MakeConstant(IntValue::make(relay_attrs->num_newaxis)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("full", "mnm.op.full", [&](const Attrs& attrs, const Array<Expr>& args) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<InitOpAttrs>();
  mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value())));
  mnm_args.push_back(
      MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
  return mnm_args;
});

MNM_OP_FROM_RELAY("strided_slice", "mnm.op.strided_slice",
                  [&](const Attrs& attrs, const Array<Expr>& args) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<StridedSliceAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->begin.value())));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->end.value())));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides.value())));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->slice_mode)));
                    return mnm_args;
                  });

MNM_GENERIC_ATTR_OP_FROM_RELAY("where", "mnm.op.where");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
