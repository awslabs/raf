/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/transform.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/transform.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/likes.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

std::vector<Value> RepeatSchema2Args(const RepeatArgs* args) {
  return {args->x};
}

std::vector<std::string> RepeatSchemaArgNames() {
  return {"x"};
}

Attrs RepeatSchema2Attrs(const RepeatArgs* args) {
  auto attrs = make_object<RepeatAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->data;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  attrs->repeats = args->repeats;
  return Attrs(attrs);
}

HashKey RepeatHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const RepeatArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->repeats;
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->data;
  }
  return key;
}

MNM_TVMJIT(Repeat, "mnm.op.repeat", RepeatArgs, RepeatSchema2Args, RepeatSchemaArgNames,
           RepeatSchema2Attrs, RepeatHasher);

std::vector<Value> TakeSchema2Args(const TakeArgs* args) {
  return {args->x, args->indices};
}

std::vector<std::string> TakeSchemaArgNames() {
  return {"x", "indices"};
}

Attrs TakeSchema2Attrs(const TakeArgs* args) {
  auto attrs = make_object<TakeAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->data;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  attrs->mode = "clip";
  return Attrs(attrs);
}

HashKey TakeHasher(const std::vector<Type>& param_types, const Type& y_type, const TakeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->data;
  }
  return key;
}

MNM_TVMJIT(Take, "mnm.op.take", TakeArgs, TakeSchema2Args, TakeSchemaArgNames, TakeSchema2Attrs,
           TakeHasher);

std::vector<Value> TakeDxSchema2Args(const TakeDxArgs* args) {
  return {args->x, args->y, args->dy, args->indices};
}

std::vector<std::string> TakeDxSchemaArgNames() {
  return {"x", "y", "dy", "indices"};
}

Attrs TakeDxSchema2Attrs(const TakeDxArgs* args) {
  auto attrs = make_object<TakeAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->data;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  attrs->mode = "wrap";
  return Attrs(attrs);
}

HashKey TakeDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const TakeDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->data;
  }
  return key;
}

MNM_TVMJIT(TakeDx, "mnm.op.take_dx", TakeDxArgs, TakeDxSchema2Args, TakeDxSchemaArgNames,
           TakeDxSchema2Attrs, TakeDxHasher);

std::vector<Value> SequenceMaskSchema2Args(const SequenceMaskArgs* args) {
  return {args->x, args->sequence_length};
}

std::vector<std::string> SequenceMaskSchemaArgNames() {
  return {"x", "sequence_length"};
}

Attrs SequenceMaskSchema2Attrs(const SequenceMaskArgs* args) {
  auto attrs = make_object<SequenceMaskAttrs>();
  attrs->mask_value = args->mask_value;
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SequenceMaskHasher(const std::vector<Type>& param_types, const Type& y_type,
                           const SequenceMaskArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->mask_value;
  key << args->axis;
  return key;
}

MNM_TVMJIT(SequenceMask, "mnm.op.sequence_mask", SequenceMaskArgs, SequenceMaskSchema2Args,
           SequenceMaskSchemaArgNames, SequenceMaskSchema2Attrs, SequenceMaskHasher);

std::vector<Value> ReverseSchema2Args(const ReverseArgs* args) {
  return {args->x};
}

std::vector<std::string> ReverseSchemaArgNames() {
  return {"x"};
}

Attrs ReverseSchema2Attrs(const ReverseArgs* args) {
  auto attrs = make_object<ReverseAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey ReverseHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ReverseArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Reverse, "mnm.op.reverse", ReverseArgs, ReverseSchema2Args, ReverseSchemaArgNames,
           ReverseSchema2Attrs, ReverseHasher);

std::vector<Value> ReverseSequenceSchema2Args(const ReverseSequenceArgs* args) {
  return {args->x, args->sequence_length};
}

std::vector<std::string> ReverseSequenceSchemaArgNames() {
  return {"x", "sequence_length"};
}

Attrs ReverseSequenceSchema2Attrs(const ReverseSequenceArgs* args) {
  auto attrs = make_object<ReverseSequenceAttrs>();
  attrs->seq_axis = args->seq_axis;
  attrs->batch_axis = args->batch_axis;
  return Attrs(attrs);
}

HashKey ReverseSequenceHasher(const std::vector<Type>& param_types, const Type& y_type,
                              const ReverseSequenceArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->seq_axis;
  key << args->batch_axis;
  return key;
}

MNM_TVMJIT(ReverseSequence, "mnm.op.reverse_sequence", ReverseSequenceArgs,
           ReverseSequenceSchema2Args, ReverseSequenceSchemaArgNames, ReverseSequenceSchema2Attrs,
           ReverseSequenceHasher);

std::vector<Value> BroadcastToSchema2Args(const BroadcastToArgs* args) {
  return {args->x};
}

std::vector<std::string> BroadcastToSchemaArgNames() {
  return {"x"};
}

Attrs BroadcastToSchema2Attrs(const BroadcastToArgs* args) {
  auto attrs = make_object<InitOpAttrs>();
  std::vector<IndexExpr> shape;
  shape.reserve(args->shape.size());
  for (size_t i = 0; i < args->shape.size(); ++i) {
    shape.emplace_back(IntImm(ir::DataType::Int(32), args->shape[i]));
  }
  attrs->shape = Array<Integer>(shape.begin(), shape.end());
  return Attrs(attrs);
}

MNM_TVMJIT(BroadcastTo, "mnm.op.broadcast_to", BroadcastToArgs, BroadcastToSchema2Args,
           BroadcastToSchemaArgNames, BroadcastToSchema2Attrs, GenericHasher);

std::vector<Value> TransposeSchema2Args(const TransposeArgs* args) {
  return {args->x};
}

std::vector<std::string> TransposeSchemaArgNames() {
  return {"x"};
}

Attrs TransposeSchema2Attrs(const TransposeArgs* args) {
  auto attrs = make_object<TransposeAttrs>();
  std::vector<Integer> axes;
  axes.reserve(args->axes.size());
  for (size_t i = 0; i < args->axes.size(); ++i) {
    axes.emplace_back(args->axes[i]);
  }
  attrs->axes = Array<Integer>(axes.begin(), axes.end());
  return Attrs(attrs);
}

HashKey TransposeHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const TransposeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axes;
  return key;
}

MNM_TVMJIT(Transpose, "mnm.op.transpose", TransposeArgs, TransposeSchema2Args,
           TransposeSchemaArgNames, TransposeSchema2Attrs, TransposeHasher);

std::vector<Value> TransposeDxSchema2Args(const TransposeDxArgs* args) {
  return {args->x, args->y, args->dy};
}

std::vector<std::string> TransposeDxSchemaArgNames() {
  return {"x", "y", "dy"};
}

Attrs TransposeDxSchema2Attrs(const TransposeDxArgs* args) {
  auto attrs = make_object<TransposeAttrs>();
  std::vector<Integer> axes;
  axes.reserve(args->axes.size());
  for (size_t i = 0; i < args->axes.size(); ++i) {
    axes.emplace_back(args->axes[i]);
  }
  attrs->axes = Array<Integer>(axes.begin(), axes.end());
  return Attrs(attrs);
}

HashKey TransposeDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const TransposeDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axes;
  return key;
}

MNM_TVMJIT(TransposeDx, "mnm.op.transpose_dx", TransposeDxArgs, TransposeDxSchema2Args,
           TransposeDxSchemaArgNames, TransposeDxSchema2Attrs, TransposeDxHasher);

std::vector<Value> BroadcastToLikeSchema2Args(const BroadcastToLikeArgs* args) {
  return {args->x, args->broadcast_type};
}

std::vector<std::string> BroadcastToLikeSchemaArgNames() {
  return {"x", "broadcast_type"};
}

Attrs BroadcastToLikeSchema2Attrs(const BroadcastToLikeArgs* args) {
  auto attrs = make_object<InitOpAttrs>();
  return Attrs(attrs);
}

MNM_TVMJIT(BroadcastToLike, "mnm.op.broadcast_to_like", BroadcastToLikeArgs,
           BroadcastToLikeSchema2Args, BroadcastToLikeSchemaArgNames, BroadcastToLikeSchema2Attrs,
           GenericHasher);

std::vector<Value> SplitSchema2Args(const SplitArgs* args) {
  return {args->x};
}

std::vector<std::string> SplitSchemaArgNames() {
  return {"x"};
}

Attrs SplitSchema2Attrs(const SplitArgs* args) {
  auto attrs = make_object<SplitAttrs>();
  value::Value indices_or_sections = args->indices_or_sections;
  // Scalar is sections, Tuple value is indices
  if (const auto* scalar = indices_or_sections.as<IntValueObj>()) {
    int64_t sections = scalar->data;
    attrs->indices_or_sections = IntImm(ir::DataType::Int(32), sections);
  } else if (const auto* tup = indices_or_sections.as<TupleValueObj>()) {
    std::vector<int64_t> indices;
    for (auto field : tup->fields) {
      auto int_value = field.as<IntValueObj>();
      indices.push_back(int_value->data);
    }
    attrs->indices_or_sections = mnm::common::shape_utils::StdVector2Array(indices);
  }

  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SplitHasher(const std::vector<Type>& param_types, const Type& y_type,
                    const SplitArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Split, "mnm.op.split", SplitArgs, SplitSchema2Args, SplitSchemaArgNames,
           SplitSchema2Attrs, SplitHasher);

std::vector<Value> ConcatenateSchema2Args(const ConcatenateArgs* args) {
  std::vector<Value> ret;
  for (auto v : args->x) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> ConcatenateSchemaArgNames() {
  return {"x"};
}

Attrs ConcatenateSchema2Attrs(const ConcatenateArgs* args) {
  auto attrs = make_object<ConcatenateAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey ConcatenateHasher(const std::vector<Type>& param_types, const Type& y_type,
                          const ConcatenateArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Concatenate, "mnm.op.concatenate", ConcatenateArgs, ConcatenateSchema2Args,
           ConcatenateSchemaArgNames, ConcatenateSchema2Attrs, ConcatenateHasher);

std::vector<Value> StackSchema2Args(const StackArgs* args) {
  std::vector<Value> ret;
  for (auto v : args->x) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> StackSchemaArgNames() {
  return {"x"};
}

Attrs StackSchema2Attrs(const StackArgs* args) {
  auto attrs = make_object<StackAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey StackHasher(const std::vector<Type>& param_types, const Type& y_type,
                    const StackArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Stack, "mnm.op.stack", StackArgs, StackSchema2Args, StackSchemaArgNames,
           StackSchema2Attrs, StackHasher);

std::vector<Value> ClipSchema2Args(const ClipArgs* args) {
  return {args->x};
}

std::vector<std::string> ClipSchemaArgNames() {
  return {"x"};
}

Attrs ClipSchema2Attrs(const ClipArgs* args) {
  auto attrs = make_object<ClipAttrs>();
  attrs->a_min = args->a_min;
  attrs->a_max = args->a_max;
  return Attrs(attrs);
}

HashKey ClipHasher(const std::vector<Type>& param_types, const Type& y_type, const ClipArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->a_min;
  key << args->a_max;
  return key;
}

MNM_TVMJIT(Clip, "mnm.op.clip", ClipArgs, ClipSchema2Args, ClipSchemaArgNames, ClipSchema2Attrs,
           ClipHasher);

std::vector<Value> ClipDxSchema2Args(const ClipDxArgs* args) {
  return {args->x, args->dy};
}

std::vector<std::string> ClipDxSchemaArgNames() {
  return {"x", "dy"};
}

Attrs ClipDxSchema2Attrs(const ClipDxArgs* args) {
  auto attrs = make_object<ClipAttrs>();
  attrs->a_min = args->a_min;
  attrs->a_max = args->a_max;
  return Attrs(attrs);
}

HashKey ClipDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const ClipDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->a_min;
  key << args->a_max;
  return key;
}

MNM_TVMJIT(ClipDx, "mnm.op.clip_dx", ClipDxArgs, ClipDxSchema2Args, ClipDxSchemaArgNames,
           ClipDxSchema2Attrs, ClipDxHasher);

std::vector<Value> CastSchema2Args(const CastArgs* args) {
  return {args->data};
}

std::vector<std::string> CastSchemaArgNames() {
  return {"data"};
}

Attrs CastSchema2Attrs(const CastArgs* args) {
  auto attrs = make_object<CastAttrs>();
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey CastHasher(const std::vector<Type>& param_types, const Type& y_type, const CastArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << ir::String2DLDataType(args->dtype);
  return key;
}

MNM_TVMJIT(Cast, "mnm.op.cast", CastArgs, CastSchema2Args, CastSchemaArgNames, CastSchema2Attrs,
           CastHasher);

std::vector<Value> CastLikeSchema2Args(const CastLikeArgs* args) {
  return {args->data, args->dtype_like};
}

std::vector<std::string> CastLikeSchemaArgNames() {
  return {"data", "dtype_like"};
}

MNM_TVMJIT(CastLike, "mnm.op.cast_like", CastLikeArgs, CastLikeSchema2Args, CastLikeSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> GatherNdSchema2Args(const GatherNdArgs* args) {
  return {args->data, args->indices};
}

std::vector<std::string> GatherNdSchemaArgNames() {
  return {"data", "indices"};
}

MNM_TVMJIT(GatherNd, "mnm.op.gather_nd", GatherNdArgs, GatherNdSchema2Args, GatherNdSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> GatherNdDxSchema2Args(const GatherNdDxArgs* args) {
  return {args->data, args->indices, args->dy};
}

std::vector<std::string> GatherNdDxSchemaArgNames() {
  return {"data", "indices", "dy"};
}

MNM_TVMJIT(GatherNdDx, "mnm.op.gather_nd_dx", GatherNdDxArgs, GatherNdDxSchema2Args,
           GatherNdDxSchemaArgNames, GenericAttrs, GenericHasher);

std::vector<Value> SqueezeSchema2Args(const SqueezeArgs* args) {
  return {args->x};
}

std::vector<std::string> SqueezeSchemaArgNames() {
  return {"x"};
}

Attrs SqueezeSchema2Attrs(const SqueezeArgs* args) {
  auto attrs = make_object<SqueezeAttrs>();
  std::vector<Integer> axis;
  axis.reserve(args->axis.size());
  for (size_t i = 0; i < args->axis.size(); ++i) {
    axis.emplace_back(args->axis[i]);
  }
  attrs->axis = Array<Integer>(axis.begin(), axis.end());
  return Attrs(attrs);
}

HashKey SqueezeHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const SqueezeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Squeeze, "mnm.op.squeeze", SqueezeArgs, SqueezeSchema2Args, SqueezeSchemaArgNames,
           SqueezeSchema2Attrs, SqueezeHasher);

std::vector<Value> ReshapeSchema2Args(const ReshapeArgs* args) {
  return {args->x};
}

std::vector<std::string> ReshapeSchemaArgNames() {
  return {"x"};
}

Attrs ReshapeSchema2Attrs(const ReshapeArgs* args) {
  auto attrs = make_object<ReshapeAttrs>();
  for (auto dim : args->shape) {
    attrs->newshape.push_back(dim);
  }
  attrs->reverse = args->reverse;
  return Attrs(attrs);
}

HashKey ReshapeHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ReshapeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  key << args->reverse;
  return key;
}

MNM_TVMJIT(Reshape, "mnm.op.reshape", ReshapeArgs, ReshapeSchema2Args, ReshapeSchemaArgNames,
           ReshapeSchema2Attrs, ReshapeHasher);

std::vector<Value> ExpandDimsSchema2Args(const ExpandDimsArgs* args) {
  return {args->x};
}

std::vector<std::string> ExpandDimsSchemaArgNames() {
  return {"x"};
}

Attrs ExpandDimsSchema2Attrs(const ExpandDimsArgs* args) {
  auto attrs = make_object<ExpandDimsAttrs>();
  attrs->axis = args->axis;
  attrs->num_newaxis = args->num_newaxis;
  return Attrs(attrs);
}

HashKey ExpandDimsHasher(const std::vector<Type>& param_types, const Type& y_type,
                         const ExpandDimsArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->num_newaxis;
  return key;
}

MNM_TVMJIT(ExpandDims, "mnm.op.expand_dims", ExpandDimsArgs, ExpandDimsSchema2Args,
           ExpandDimsSchemaArgNames, ExpandDimsSchema2Attrs, ExpandDimsHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
