/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/transform.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <vector>
#include <raf/op_utils.h>
#include <raf/cache.h>
#include <raf/value.h>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/likes.h"
#include "../../schema/nn.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;

Attrs ArangeSchema2Attrs(const ArangeArgs* args) {
  auto attrs = make_object<ArangeAttrs>();
  attrs->start = MakeConstant(args->start);
  attrs->stop = MakeConstant(args->stop);
  attrs->step = MakeConstant(args->step);
  attrs->dtype = DataType(String2DLDataType(args->dtype));
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

RAF_TVM(arange, Arange, ArangeArgs, ArangeSchema2Args, ArangeSchemaArgNames, ArangeSchema2Attrs,
        ArangeHasher, kOpaque);

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

RAF_TVM(adv_index, AdvIndex, AdvIndexArgs, AdvIndexSchema2Args, AdvIndexSchemaArgNames,
        GenericAttrs, GenericHasher, kOpaque);

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

RAF_TVM(adv_index_dx, AdvIndexDx, AdvIndexDxArgs, AdvIndexDxSchema2Args, AdvIndexDxSchemaArgNames,
        GenericAttrs, GenericHasher, kOpaque);

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

RAF_TVM(repeat, Repeat, RepeatArgs, RepeatSchema2Args, RepeatSchemaArgNames, RepeatSchema2Attrs,
        RepeatHasher, kBroadcast);

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

RAF_TVM(repeat_dx, RepeatDx, RepeatDxArgs, RepeatDxSchema2Args, RepeatDxSchemaArgNames,
        RepeatDxSchema2Attrs, RepeatDxHasher, kCommReduce);

template <typename T>
std::vector<Value> TakeSchema2Args(const T* args) {
  return {args->x, args->indices};
}

std::vector<std::string> TakeSchemaArgNames(const op::CallValues& call) {
  return {"x", "indices"};
}

Attrs TakeSchema2Attrs(const TakeArgs* args) {
  auto attrs = make_object<TakeAttrs>();
  attrs->batch_dims = 0;
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

RAF_TVM(take, Take, TakeArgs, TakeSchema2Args<TakeArgs>, TakeSchemaArgNames, TakeSchema2Attrs,
        TakeHasher, kInjective);

RAF_TVM(embedding, Embedding, EmbeddingArgs, TakeSchema2Args<EmbeddingArgs>, TakeSchemaArgNames,
        GenericAttrs, GenericHasher, kInjective);

std::vector<Value> TakeDxSchema2Args(const TakeDxArgs* args) {
  return {args->x, args->dy, args->indices};
}

std::vector<std::string> TakeDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "dy", "indices"};
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

RAF_TVM(take_dx, TakeDx, TakeDxArgs, TakeDxSchema2Args, TakeDxSchemaArgNames, TakeDxSchema2Attrs,
        TakeDxHasher, kOpaque);

std::vector<Value> EmbeddingDxSchema2Args(const EmbeddingDxArgs* args) {
  return {args->dy, args->indices};
}

std::vector<std::string> EmbeddingDxSchemaArgNames(const op::CallValues& call) {
  return {"dy", "indices"};
}

Attrs EmbeddingDxSchema2Attrs(const EmbeddingDxArgs* args) {
  auto attrs = make_object<DimAttrs>();
  auto num_weight = GetShapeVecFromValue(args->num_weight);
  for (auto v : num_weight) {
    attrs->dims.push_back(Integer(v));
  }
  return Attrs(attrs);
}

RAF_TVM(embedding_dx, EmbeddingDx, EmbeddingDxArgs, EmbeddingDxSchema2Args,
        EmbeddingDxSchemaArgNames, EmbeddingDxSchema2Attrs, GenericHasher, kOpaque);

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

RAF_TVM(sequence_mask, SequenceMask, SequenceMaskArgs, SequenceMaskSchema2Args,
        SequenceMaskSchemaArgNames, SequenceMaskSchema2Attrs, SequenceMaskHasher, kInjective);

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

RAF_TVM(reverse, Reverse, ReverseArgs, ReverseSchema2Args, ReverseSchemaArgNames,
        ReverseSchema2Attrs, ReverseHasher, kInjective);

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

RAF_TVM(reverse_sequence, ReverseSequence, ReverseSequenceArgs, ReverseSequenceSchema2Args,
        ReverseSequenceSchemaArgNames, ReverseSequenceSchema2Attrs, ReverseSequenceHasher,
        kInjective);

std::vector<Value> BinaryToSchema2Args(const BinaryToArgs* args) {
  return {args->x};
}

std::vector<std::string> BinaryToSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

RAF_TVM(broadcast_to, BroadcastTo, BinaryToArgs, BinaryToSchema2Args, BinaryToSchemaArgNames,
        GenericAttrs, GenericHasher, kBroadcast);

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

RAF_TVM(transpose, Transpose, TransposeArgs, TransposeSchema2Args, TransposeSchemaArgNames,
        TransposeSchema2Attrs, TransposeHasher, kInjective);

RAF_TVM(transpose_dx, TransposeDx, TransposeArgs, TransposeSchema2Args, TransposeSchemaArgNames,
        TransposeSchema2Attrs, TransposeHasher, kInjective);

std::vector<Value> BinaryLikeSchema2Args(const BinaryLikeArgs* args) {
  return {args->x, args->like_type};
}

std::vector<std::string> BinaryLikeSchemaArgNames(const op::CallValues& call) {
  return {"x", "like_type"};
}

RAF_TVM(broadcast_to_like, BroadcastToLike, BinaryLikeArgs, BinaryLikeSchema2Args,
        BinaryLikeSchemaArgNames, GenericAttrs, GenericHasher, kBroadcast);

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
      indices.push_back(GetScalarValueData<int64_t>(field));
    }
    attrs->indices_or_sections = raf::common::shape_utils::StdVector2Array(indices);
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

RAF_TVM(split, Split, SplitArgs, SplitSchema2Args, SplitSchemaArgNames, SplitSchema2Attrs,
        SplitHasher, kInjective);

std::vector<Value> ScatterSchema2Args(const ScatterArgs* args) {
  return {args->x, args->index, args->src};
}

std::vector<std::string> ScatterSchemaArgNames(const op::CallValues& call) {
  return {"x", "index", "src"};
}

Attrs ScatterSchema2Attrs(const ScatterArgs* args) {
  auto attrs = make_object<ScatterAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  return Attrs(attrs);
}

HashKey ScatterHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ScatterArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->value;
  }
  return key;
}

RAF_TVM(scatter, Scatter, ScatterArgs, ScatterSchema2Args, ScatterSchemaArgNames,
        ScatterSchema2Attrs, ScatterHasher, kInjective);

std::vector<Value> ScatterDxSchema2Args(const ScatterDxArgs* args) {
  return {args->x, args->y, args->dy, args->index, args->src};
}

std::vector<std::string> ScatterDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "y", "dy", "index", "src"};
}

Attrs ScatterDxSchema2Attrs(const ScatterDxArgs* args) {
  auto attrs = make_object<ScatterAttrs>();
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    attrs->axis = v->value;
  } else {
    attrs->axis = NullValue<Integer>();
  }
  return Attrs(attrs);
}

HashKey ScatterDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const ScatterDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->value;
  }
  return key;
}

RAF_TVM(scatter_dx, ScatterDx, ScatterDxArgs, ScatterDxSchema2Args, ScatterDxSchemaArgNames,
        ScatterDxSchema2Attrs, ScatterDxHasher, kInjective);

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

RAF_TVM(concatenate, Concatenate, ConcatenateArgs, ConcatenateSchema2Args,
        ConcatenateSchemaArgNames, ConcatenateSchema2Attrs, ConcatenateHasher, kInjective);

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

RAF_TVM(mesh_grid, MeshGrid, MeshGridArgs, MeshGridSchema2Args, MeshGridSchemaArgNames,
        GenericAttrs, GenericHasher, kInjective);

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

RAF_TVM(stack, Stack, StackArgs, StackSchema2Args, StackSchemaArgNames, StackSchema2Attrs,
        StackHasher, kInjective);

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

RAF_TVM(clip, Clip, ClipArgs, ClipSchema2Args, ClipSchemaArgNames, ClipSchema2Attrs, ClipHasher,
        kElemWise);

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

RAF_TVM(clip_dx, ClipDx, ClipDxArgs, ClipDxSchema2Args, ClipDxSchemaArgNames, ClipDxSchema2Attrs,
        ClipDxHasher, kElemWise);

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

RAF_TVM(cast, Cast, CastArgs, CastSchema2Args, CastSchemaArgNames, CastSchema2Attrs, CastHasher,
        kElemWise);

RAF_TVM(cast_like, CastLike, BinaryLikeArgs, BinaryLikeSchema2Args, BinaryLikeSchemaArgNames,
        GenericAttrs, GenericHasher, kElemWise);

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

RAF_TVM(gather, Gather, GatherArgs, GatherSchema2Args, GatherSchemaArgNames, GatherSchema2Attrs,
        GatherHasher, kInjective);

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

RAF_TVM(gather_dx, GatherDx, GatherDxArgs, GatherDxSchema2Args, GatherDxSchemaArgNames,
        GatherDxSchema2Attrs, GatherDxHasher, kInjective);

std::vector<Value> GatherNdSchema2Args(const GatherNdArgs* args) {
  return {args->data, args->indices};
}

std::vector<std::string> GatherNdSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices"};
}

Attrs GatherNdSchema2Attrs(const GatherNdArgs* args) {
  auto attrs = make_object<GatherNDAttrs>();
  attrs->batch_dims = 0;
  attrs->index_rank = NullValue<Integer>();
  return Attrs(attrs);
}

RAF_TVM(gather_nd, GatherNd, GatherNdArgs, GatherNdSchema2Args, GatherNdSchemaArgNames,
        GatherNdSchema2Attrs, GenericHasher, kInjective);

std::vector<Value> GatherNdDxSchema2Args(const GatherNdDxArgs* args) {
  return {args->data, args->indices, args->dy};
}

std::vector<std::string> GatherNdDxSchemaArgNames(const op::CallValues& call) {
  return {"data", "indices", "dy"};
}

RAF_TVM(gather_nd_dx, GatherNdDx, GatherNdDxArgs, GatherNdDxSchema2Args, GatherNdDxSchemaArgNames,
        GenericAttrs, GenericHasher, kInjective);

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

RAF_TVM(squeeze, Squeeze, SqueezeArgs, SqueezeSchema2Args, SqueezeSchemaArgNames,
        SqueezeSchema2Attrs, SqueezeHasher, kInjective);

std::vector<Value> ReshapeSchema2Args(const ReshapeArgs* args) {
  return {args->x};
}

std::vector<std::string> ReshapeSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ReshapeSchema2Attrs(const ReshapeArgs* args) {
  auto attrs = make_object<ReshapeAttrs>();
  std::vector<int64_t> shape_vec = GetShapeVecFromValue(args->shape);
  Array<Integer> newshape;
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    newshape.push_back(shape_vec[i]);
  }
  attrs->newshape = newshape;
  // FIXME(comaniac): attrs->reverse has been removed on Relay side so we get rid of
  // that attribute here. It might be an issue with reshape(shape, reverse=True).
  CHECK(!args->reverse);
  return Attrs(attrs);
}

HashKey ReshapeHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ReshapeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  key << shape;
  key << args->reverse;
  return key;
}

RAF_TVM(reshape, Reshape, ReshapeArgs, ReshapeSchema2Args, ReshapeSchemaArgNames,
        ReshapeSchema2Attrs, ReshapeHasher, kInjective);

template <typename T>
std::vector<Value> ResizeSchema2Args(const T* args) {
  return {args->x};
}

std::vector<std::string> ResizeSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs Resize2DSchema2Attrs(const Resize2DArgs* args) {
  auto attrs = make_object<Resize2DAttrs>();
  attrs->layout = args->layout;
  attrs->method = args->method;
  attrs->coordinate_transformation_mode = args->coordinate_transformation_mode;
  attrs->rounding_method = args->rounding_method;
  attrs->cubic_alpha = args->cubic_alpha;
  attrs->cubic_exclude = args->cubic_exclude;

  DataType out_dtype(String2DLDataType(args->out_dtype));
  attrs->out_dtype = out_dtype;

  std::vector<int64_t> size = GetShapeVecFromValue(args->size);
  CHECK(size.size() > 0);
  if (size.size() == 1) size.push_back(size[0]);

  for (auto dim : size) {
    attrs->size.push_back(IndexExpr((int32_t)dim));
  }
  return Attrs(attrs);
}

template <typename T>
HashKey ResizeHasher(const std::vector<Type>& param_types, const Type& y_type, const T* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << GetShapeVecFromValue(args->size);
  key << args->layout;
  key << args->method;
  key << args->coordinate_transformation_mode;
  key << args->rounding_method;
  key << args->cubic_alpha;
  key << args->cubic_exclude;
  key << args->out_dtype;
  return key;
}

RAF_TVM(resize2d, Resize2D, Resize2DArgs, ResizeSchema2Args<Resize2DArgs>, ResizeSchemaArgNames,
        Resize2DSchema2Attrs, ResizeHasher<Resize2DArgs>, kInjective);

std::vector<Value> Resize2DDxSchema2Args(const Resize2DDxArgs* args) {
  return {args->x, args->dy};
}

std::vector<std::string> Resize2DDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "dy"};
}

Attrs Resize2DDxSchema2Attrs(const Resize2DDxArgs* args) {
  auto attrs = make_object<Resize2DAttrs>();
  attrs->layout = args->layout;
  attrs->method = args->method;
  attrs->coordinate_transformation_mode = args->coordinate_transformation_mode;
  attrs->rounding_method = args->rounding_method;
  attrs->cubic_alpha = args->cubic_alpha;
  attrs->cubic_exclude = args->cubic_exclude;

  DataType out_dtype(String2DLDataType(args->out_dtype));
  attrs->out_dtype = out_dtype;

  std::vector<int64_t> size(args->size);
  CHECK(size.size() > 0);
  if (size.size() == 1) size.push_back(size[0]);

  for (auto dim : size) {
    attrs->size.push_back(IndexExpr((int32_t)dim));
  }

  return Attrs(attrs);
}

HashKey Resize2DDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                         const Resize2DDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->size;
  key << args->layout;
  key << args->method;
  key << args->coordinate_transformation_mode;
  key << args->rounding_method;
  key << args->cubic_alpha;
  key << args->cubic_exclude;
  key << args->out_dtype;
  return key;
}

RAF_TVM(resize2d_dx, Resize2DDx, Resize2DDxArgs, Resize2DDxSchema2Args, Resize2DDxSchemaArgNames,
        Resize2DDxSchema2Attrs, Resize2DDxHasher, kInjective);

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

RAF_TVM(expand_dims, ExpandDims, ExpandDimsArgs, ExpandDimsSchema2Args, ExpandDimsSchemaArgNames,
        ExpandDimsSchema2Attrs, ExpandDimsHasher, kBroadcast);

std::vector<Value> FullSchema2Args(const FullArgs* args) {
  return {};
}

std::vector<std::string> FullSchemaArgNames(const op::CallValues& call) {
  return {};
}

Attrs FullSchema2Attrs(const FullArgs* args) {
  auto attrs = make_object<FullAttrs>();
  std::vector<int64_t> shape_vec = GetShapeVecFromValue(args->shape);
  std::vector<Integer> shape;
  shape.reserve(shape_vec.size());
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    shape.emplace_back(shape_vec[i]);
  }

  attrs->shape = Array<Integer>(shape.begin(), shape.end());
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  attrs->fill_value = args->fill_value;
  return Attrs(attrs);
}

HashKey FullHasher(const std::vector<Type>& param_types, const Type& y_type, const FullArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << GetShapeVecFromValue(args->shape);
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  key << args->fill_value;
  return key;
}

RAF_TVM(full, Full, FullArgs, FullSchema2Args, FullSchemaArgNames, FullSchema2Attrs, FullHasher,
        kInjective);

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

RAF_TVM(full_like, FullLike, FullLikeArgs, FullLikeSchema2Args, FullLikeSchemaArgNames,
        FullLikeSchema2Attrs, FullLikeHasher, kInjective);

std::vector<Value> StridedSliceSchema2Args(const StridedSliceArgs* args) {
  return {args->x};
}

std::vector<std::string> StridedSliceSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs StridedSliceSchema2Attrs(const StridedSliceArgs* args) {
  auto attrs = make_object<StridedSliceAttrs>();
  std::vector<int64_t> begin_imms = GetShapeVecFromValue(args->begin);
  std::vector<int64_t> end_imms = GetShapeVecFromValue(args->end);
  CHECK_EQ(begin_imms.size(), end_imms.size());
  CHECK_EQ(begin_imms.size(), args->strides.size());
  std::vector<Integer> begin, end, strides;
  for (int i = 0; i < begin_imms.size(); ++i) {
    begin.emplace_back(begin_imms[i]);
    end.emplace_back(end_imms[i]);
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
  key << GetShapeVecFromValue(args->begin);
  key << GetShapeVecFromValue(args->end);
  key << args->strides;
  key << args->slice_mode;
  return key;
}

RAF_TVM(strided_slice, StridedSlice, StridedSliceArgs, StridedSliceSchema2Args,
        StridedSliceSchemaArgNames, StridedSliceSchema2Attrs, StridedSliceHasher, kInjective);

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
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  for (int i = 0; i < shape.size(); ++i) {
    primal_shape.emplace_back(shape[i]);
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

RAF_TVM(strided_slice_dx, StridedSliceDx, StridedSliceDxArgs, StridedSliceDxSchema2Args,
        StridedSliceDxSchemaArgNames, StridedSliceDxSchema2Attrs, StridedSliceDxHasher, kInjective);

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

// FIXME: where should be kBroadcast, but it might be super slow when fused with other ops
// such as sum. We should change it back to kBroadcast after resolving this issue.
RAF_TVM(where, Where, WhereArgs, WhereSchema2Args, WhereSchemaArgNames, GenericAttrs, GenericHasher,
        kOutEWiseFusable);

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

RAF_TVM(swap_axis, SwapAxis, SwapAxisArgs, SwapAxisSchema2Args, SwapAxisSchemaArgNames,
        SwapAxisSchema2Attrs, SwapAxisHasher, kInjective);

std::vector<Value> ArgwhereSchema2Args(const ArgwhereArgs* args) {
  return {args->condition};
}

std::vector<std::string> ArgwhereSchemaArgNames(const op::CallValues& call) {
  return {"condition"};
}

RAF_TVM(upper_bound.argwhere, Argwhere, ArgwhereArgs, ArgwhereSchema2Args, ArgwhereSchemaArgNames,
        GenericAttrs, GenericHasher, kOpaque);

std::vector<Value> CumsumSchema2Args(const CumsumArgs* args) {
  return {args->x};
}

std::vector<std::string> CumsumSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs CumsumSchema2Attrs(const CumsumArgs* args) {
  auto attrs = make_object<tvm::relay::ScanopAttrs>();
  attrs->axis = Integer(args->axis);
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  attrs->exclusive = Bool(args->exclusive);
  return Attrs(attrs);
}

HashKey CumsumHasher(const std::vector<Type>& param_types, const Type& ret_type,
                     const CumsumArgs* args) {
  HashKey key = GenericHasher<std::nullptr_t>(param_types, ret_type, nullptr);
  key << args->axis;
  key << args->dtype;
  key << args->exclusive;
  return key;
}

RAF_TVM(cumsum, Cumsum, CumsumArgs, CumsumSchema2Args, CumsumSchemaArgNames, CumsumSchema2Attrs,
        CumsumHasher, kOpaque);

RAF_TVM(collapse_sum_like, CollapseSumLike, BinaryLikeArgs, BinaryLikeSchema2Args,
        BinaryLikeSchemaArgNames, GenericAttrs, GenericHasher, kCommReduce);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
