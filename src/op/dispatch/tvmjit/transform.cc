/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/transform.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/transform.h>
#include <mnm/tvmjit/transform.h>
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

Attrs ArangeSchema2Attrs(const ArangeArgs* args) {
  auto attrs = make_object<ArangeAttrs>();
  attrs->start = MakeConstant(args->start);
  attrs->stop = MakeConstant(args->stop);
  attrs->step = MakeConstant(args->step);
  attrs->dtype = runtime::DataType(String2DLDataType(args->dtype));
  return Attrs(attrs);
}

std::vector<Value> ArangeSchema2Args(const ArangeArgs* args) {
  std::vector<Value> out;
  out.push_back(args->start);
  out.push_back(args->stop);
  out.push_back(args->step);
  return out;
}

std::vector<std::string> ArangeSchemaArgNames(const op::CallValues& call) {
  return {"start", "stop", "step"};
}

HashKey ArangeHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const ArangeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->start;
  key << args->stop;
  key << args->step;
  return key;
}

MNM_TVMJIT(Arange, "mnm.op.arange", ArangeArgs, ArangeSchema2Args, ArangeSchemaArgNames,
           ArangeSchema2Attrs, ArangeHasher);

std::vector<Value> AdvIndexSchema2Args(const AdvIndexArgs* args) {
  std::vector<Value> ret;
  for (auto v : args->inputs) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> AdvIndexSchemaArgNames(const op::CallValues& call) {
  return {"inputs"};
}

MNM_TVMJIT(AdvIndex, "mnm.op.adv_index", AdvIndexArgs, AdvIndexSchema2Args, AdvIndexSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> AdvIndexDxSchema2Args(const AdvIndexDxArgs* args) {
  std::vector<Value> ret;
  ret.push_back(args->dy);
  for (auto v : args->inputs) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> AdvIndexDxSchemaArgNames(const op::CallValues& call) {
  return {"dy", "inputs"};
}

MNM_TVMJIT(AdvIndexDx, "mnm.op.adv_index_dx", AdvIndexDxArgs, AdvIndexDxSchema2Args,
           AdvIndexDxSchemaArgNames, GenericAttrs, GenericHasher);

std::vector<Value> RepeatSchema2Args(const RepeatArgs* args) {
  return {args->x};
}

std::vector<std::string> RepeatSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs RepeatSchema2Attrs(const RepeatArgs* args) {
  auto attrs = make_object<RepeatAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
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
    key << v->value;
  }
  return key;
}

MNM_TVMJIT(Repeat, "mnm.op.repeat", RepeatArgs, RepeatSchema2Args, RepeatSchemaArgNames,
           RepeatSchema2Attrs, RepeatHasher);

std::vector<Value> TakeSchema2Args(const TakeArgs* args) {
  return {args->x, args->indices};
}

std::vector<std::string> TakeSchemaArgNames(const op::CallValues& call) {
  return {"x", "indices"};
}

Attrs TakeSchema2Attrs(const TakeArgs* args) {
  auto attrs = make_object<TakeAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  attrs->mode = args->mode;
  return Attrs(attrs);
}

HashKey TakeHasher(const std::vector<Type>& param_types, const Type& y_type, const TakeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->value;
  }
  key << args->mode;
  return key;
}

MNM_TVMJIT(Take, "mnm.op.take", TakeArgs, TakeSchema2Args, TakeSchemaArgNames, TakeSchema2Attrs,
           TakeHasher);

std::vector<Value> TakeDxSchema2Args(const TakeDxArgs* args) {
  return {args->x, args->y, args->dy, args->indices};
}

std::vector<std::string> TakeDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "y", "dy", "indices"};
}

Attrs TakeDxSchema2Attrs(const TakeDxArgs* args) {
  auto attrs = make_object<TakeAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  attrs->mode = args->mode;
  return Attrs(attrs);
}

HashKey TakeDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const TakeDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->value;
  }
  key << args->mode;
  return key;
}

MNM_TVMJIT(TakeDx, "mnm.op.take_dx", TakeDxArgs, TakeDxSchema2Args, TakeDxSchemaArgNames,
           TakeDxSchema2Attrs, TakeDxHasher);

std::vector<Value> SequenceMaskSchema2Args(const SequenceMaskArgs* args) {
  return {args->x, args->sequence_length};
}

std::vector<std::string> SequenceMaskSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> ReverseSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> ReverseSequenceSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> BroadcastToSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> TransposeSchemaArgNames(const op::CallValues& call) {
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
  return {args->dy};
}

std::vector<std::string> TransposeDxSchemaArgNames(const op::CallValues& call) {
  return {"dy"};
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

std::vector<Value> RepeatDxSchema2Args(const RepeatDxArgs* args) {
  return {args->x, args->dy};
}

std::vector<std::string> RepeatDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "dy"};
}

Attrs RepeatDxSchema2Attrs(const RepeatDxArgs* args) {
  auto attrs = make_object<RepeatAttrs>();
  attrs->repeats = args->repeats;
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  return Attrs(attrs);
}

HashKey RepeatDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const RepeatDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->repeats;
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->value;
  }
  return key;
}

MNM_TVMJIT(RepeatDx, "mnm.op.repeat_dx", RepeatDxArgs, RepeatDxSchema2Args, RepeatDxSchemaArgNames,
           RepeatDxSchema2Attrs, RepeatDxHasher);

std::vector<Value> BroadcastToLikeSchema2Args(const BroadcastToLikeArgs* args) {
  return {args->x, args->broadcast_type};
}

std::vector<std::string> BroadcastToLikeSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> SplitSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SplitSchema2Attrs(const SplitArgs* args) {
  auto attrs = make_object<SplitAttrs>();
  value::Value indices_or_sections = args->indices_or_sections;
  // Scalar is sections, Tuple value is indices
  if (const auto* scalar = indices_or_sections.as<IntValueObj>()) {
    int64_t sections = scalar->value;
    attrs->indices_or_sections = IntImm(ir::DataType::Int(32), sections);
  } else if (const auto* tup = indices_or_sections.as<TupleValueObj>()) {
    std::vector<int64_t> indices;
    for (auto field : tup->fields) {
      auto int_value = field.as<IntValueObj>();
      indices.push_back(int_value->value);
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

std::vector<std::string> ConcatenateSchemaArgNames(const op::CallValues& call) {
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

std::vector<Value> MeshGridSchema2Args(const MeshGridArgs* args) {
  std::vector<Value> ret;
  for (auto v : args->x) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> MeshGridSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

MNM_TVMJIT(MeshGrid, "mnm.op.mesh_grid", MeshGridArgs, MeshGridSchema2Args, MeshGridSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> StackSchema2Args(const StackArgs* args) {
  std::vector<Value> ret;
  for (auto v : args->x) {
    ret.push_back(v);
  }
  return ret;
}

std::vector<std::string> StackSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> ClipSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> ClipDxSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> CastSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> CastLikeSchemaArgNames(const op::CallValues& call) {
  return {"data", "dtype_like"};
}

MNM_TVMJIT(CastLike, "mnm.op.cast_like", CastLikeArgs, CastLikeSchema2Args, CastLikeSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> GatherSchema2Args(const GatherArgs* args) {
  return {args->data, args->indices};
}

std::vector<std::string> GatherSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices"};
}

Attrs GatherSchema2Attrs(const GatherArgs* args) {
  auto attrs = make_object<GatherAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey GatherHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const GatherArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(Gather, "mnm.op.gather", GatherArgs, GatherSchema2Args, GatherSchemaArgNames,
           GatherSchema2Attrs, GatherHasher);

std::vector<Value> GatherDxSchema2Args(const GatherDxArgs* args) {
  return {args->data, args->indices, args->dy};
}

std::vector<std::string> GatherDxSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices", "dy"};
}

Attrs GatherDxSchema2Attrs(const GatherDxArgs* args) {
  auto attrs = make_object<GatherAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey GatherDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const GatherDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

MNM_TVMJIT(GatherDx, "mnm.op.gather_dx", GatherDxArgs, GatherDxSchema2Args, GatherDxSchemaArgNames,
           GatherDxSchema2Attrs, GatherDxHasher);

std::vector<Value> GatherNdSchema2Args(const GatherNdArgs* args) {
  return {args->data, args->indices};
}

std::vector<std::string> GatherNdSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices"};
}

MNM_TVMJIT(GatherNd, "mnm.op.gather_nd", GatherNdArgs, GatherNdSchema2Args, GatherNdSchemaArgNames,
           GenericAttrs, GenericHasher);

std::vector<Value> GatherNdDxSchema2Args(const GatherNdDxArgs* args) {
  return {args->data, args->indices, args->dy};
}

std::vector<std::string> GatherNdDxSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices", "dy"};
}

MNM_TVMJIT(GatherNdDx, "mnm.op.gather_nd_dx", GatherNdDxArgs, GatherNdDxSchema2Args,
           GatherNdDxSchemaArgNames, GenericAttrs, GenericHasher);

std::vector<Value> SqueezeSchema2Args(const SqueezeArgs* args) {
  return {args->x};
}

std::vector<std::string> SqueezeSchemaArgNames(const op::CallValues& call) {
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

std::vector<std::string> ReshapeSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ReshapeSchema2Attrs(const ReshapeArgs* args) {
  auto attrs = make_object<ReshapeAttrs>();
  for (auto dim : args->shape) {
    attrs->newshape.push_back(dim);
  }
  // FIXME(comaniac): attrs->reverse has been removed on Relay side so we get rid of
  // that attribute here. It might be an issue with reshape(shape, reverse=True).
  CHECK(!args->reverse);
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

std::vector<std::string> ExpandDimsSchemaArgNames(const op::CallValues& call) {
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

/*! \brief Attributes that specify a tensor */
struct FullAttrs : public tvm::AttrsNode<FullAttrs> {
  Optional<Array<Integer>> shape;
  DataType dtype;
  double fill_value;

  TVM_DECLARE_ATTRS(FullAttrs, "relay.attrs.FullAttrs") {
    TVM_ATTR_FIELD(shape).describe("Target shape.");
    TVM_ATTR_FIELD(dtype).describe("Target data type.").set_default(NullValue<DataType>());
    TVM_ATTR_FIELD(fill_value).describe("Filled value.");
  }
};  // struct FullAttrs
TVM_REGISTER_NODE_TYPE(FullAttrs);

std::vector<Value> FullSchema2Args(const FullArgs* args) {
  return {};
}

std::vector<std::string> FullSchemaArgNames(const op::CallValues& call) {
  return {};
}

Attrs FullSchema2Attrs(const FullArgs* args) {
  auto attrs = make_object<FullAttrs>();
  std::vector<Integer> shape;
  shape.reserve(args->shape.size());
  for (size_t i = 0; i < args->shape.size(); ++i) {
    shape.emplace_back(args->shape[i]);
  }

  attrs->shape = Array<Integer>(shape.begin(), shape.end());
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  attrs->fill_value = args->fill_value;
  return Attrs(attrs);
}

HashKey FullHasher(const std::vector<Type>& param_types, const Type& y_type, const FullArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  key << args->fill_value;
  return key;
}

MNM_TVMJIT(Full, "mnm.op.full", FullArgs, FullSchema2Args, FullSchemaArgNames, FullSchema2Attrs,
           FullHasher);

std::vector<Value> FullLikeSchema2Args(const FullLikeArgs* args) {
  return {args->data};
}

std::vector<std::string> FullLikeSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs FullLikeSchema2Attrs(const FullLikeArgs* args) {
  auto attrs = make_object<FullAttrs>();
  attrs->fill_value = args->fill_value;
  return Attrs(attrs);
}

HashKey FullLikeHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const FullLikeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->fill_value;
  return key;
}

MNM_TVMJIT(FullLike, "mnm.op.full_like", FullLikeArgs, FullLikeSchema2Args, FullLikeSchemaArgNames,
           FullLikeSchema2Attrs, FullLikeHasher);

std::vector<Value> StridedSliceSchema2Args(const StridedSliceArgs* args) {
  return {args->x};
}

std::vector<std::string> StridedSliceSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs StridedSliceSchema2Attrs(const StridedSliceArgs* args) {
  using namespace tvm;
  auto attrs = make_object<StridedSliceAttrs>();
  CHECK_EQ(args->begin.size(), args->end.size());
  CHECK_EQ(args->begin.size(), args->strides.size());
  std::vector<Integer> begin, end, strides;
  for (int i = 0; i < args->begin.size(); ++i) {
    begin.emplace_back(args->begin[i]);
    end.emplace_back(args->end[i]);
    strides.emplace_back(args->strides[i]);
  }
  attrs->begin = Array<Integer>(begin.begin(), begin.end());
  attrs->end = Array<Integer>(end.begin(), end.end());
  attrs->strides = Array<Integer>(strides.begin(), strides.end());
  attrs->slice_mode = args->slice_mode;
  return Attrs(attrs);
}

HashKey StridedSliceHasher(const std::vector<Type>& param_types, const Type& y_type,
                           const StridedSliceArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->begin;
  key << args->end;
  key << args->strides;
  key << args->slice_mode;
  return key;
}

MNM_TVMJIT(StridedSlice, "mnm.op.strided_slice", StridedSliceArgs, StridedSliceSchema2Args,
           StridedSliceSchemaArgNames, StridedSliceSchema2Attrs, StridedSliceHasher);

std::vector<Value> StridedSliceDxSchema2Args(const StridedSliceDxArgs* args) {
  return {args->dy};
}

std::vector<std::string> StridedSliceDxSchemaArgNames(const op::CallValues& call) {
  return {"dy"};
}

Attrs StridedSliceDxSchema2Attrs(const StridedSliceDxArgs* args) {
  auto attrs = make_object<StridedSliceDxAttrs>();
  CHECK_EQ(args->begin.size(), args->end.size());
  CHECK_EQ(args->begin.size(), args->strides.size());
  std::vector<Integer> primal_shape, begin, end, strides;
  for (int i = 0; i < args->begin.size(); ++i) {
    begin.emplace_back(args->begin[i]);
    end.emplace_back(args->end[i]);
    strides.emplace_back(args->strides[i]);
  }
  for (int i = 0; i < args->primal_shape.size(); ++i) {
    primal_shape.emplace_back(args->primal_shape[i]);
  }
  attrs->primal_shape = Array<Integer>(primal_shape.begin(), primal_shape.end());
  attrs->begin = Array<Integer>(begin.begin(), begin.end());
  attrs->end = Array<Integer>(end.begin(), end.end());
  attrs->strides = Array<Integer>(strides.begin(), strides.end());
  attrs->slice_mode = args->slice_mode;
  return Attrs(attrs);
}

HashKey StridedSliceDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const StridedSliceDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->begin;
  key << args->end;
  key << args->strides;
  key << args->slice_mode;
  return key;
}

MNM_TVMJIT(StridedSliceDx, "mnm.op.strided_slice_dx", StridedSliceDxArgs, StridedSliceDxSchema2Args,
           StridedSliceDxSchemaArgNames, StridedSliceDxSchema2Attrs, StridedSliceDxHasher);

TVM_REGISTER_NODE_TYPE(StridedSliceDxAttrs);

std::vector<Value> WhereSchema2Args(const WhereArgs* args) {
  return {args->condition, args->x, args->y};
}

std::vector<std::string> WhereSchemaArgNames(const op::CallValues& call) {
  return {"condition", "x", "y"};
}

Attrs WhereSchema2Attrs(const WhereArgs* args) {
  auto attrs = make_object<InitOpAttrs>();
  return Attrs(attrs);
}

HashKey WhereHasher(const std::vector<Type>& param_types, const Type& y_type,
                    const WhereArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->condition;
  key << args->x;
  key << args->y;
  return key;
}

MNM_TVMJIT(Where, "mnm.op.where", WhereArgs, WhereSchema2Args, WhereSchemaArgNames,
           WhereSchema2Attrs, WhereHasher);

std::vector<Value> WhereDxSchema2Args(const BinaryDxArgs* args) {
  return {args->x1, args->x2, args->y, args->dy};
}

std::vector<std::string> WhereDxSchemaArgNames(const op::CallValues& call) {
  return {"x1", "x2", "y", "dy"};
}

HashKey WhereDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const BinaryDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->x1;
  key << args->x2;
  key << args->y;
  key << args->dy;
  return key;
}

MNM_TVMJIT(WhereDx, "mnm.op.where_dx", BinaryDxArgs, WhereDxSchema2Args, WhereDxSchemaArgNames,
           GenericAttrs, WhereDxHasher);

struct SwapAxisAttrs : public tvm::AttrsNode<SwapAxisAttrs> {
  int axis1;
  int axis2;
  TVM_DECLARE_ATTRS(SwapAxisAttrs, "attrs.SwapAxisAttrs") {
    TVM_ATTR_FIELD(axis1);
    TVM_ATTR_FIELD(axis2);
  }
};
TVM_REGISTER_NODE_TYPE(SwapAxisAttrs);

std::vector<Value> SwapAxisSchema2Args(const SwapAxisArgs* args) {
  return {args->x};
}

std::vector<std::string> SwapAxisSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SwapAxisSchema2Attrs(const SwapAxisArgs* args) {
  auto attrs = make_object<SwapAxisAttrs>();
  attrs->axis1 = args->axis1;
  attrs->axis2 = args->axis2;
  return Attrs(attrs);
}

HashKey SwapAxisHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const SwapAxisArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis1;
  key << args->axis2;
  return key;
}

MNM_TVMJIT(SwapAxis, "mnm.op.swap_axis", SwapAxisArgs, SwapAxisSchema2Args, SwapAxisSchemaArgNames,
           SwapAxisSchema2Attrs, SwapAxisHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
