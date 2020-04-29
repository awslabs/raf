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
#include "../../schema/nn.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace ::tvm::relay;

Attrs TakeNormalizer(TVMOpEnv* env, const TakeArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(2);
  env->inputs[0] = GetDLTensor(args->x);
  env->inputs[1] = GetDLTensor(args->indices);
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

void TakeTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  y_type[0] = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0]),
                  GetTensorType(env->inputs[1])};
}

HashKey TakeHasher(const std::vector<Type>& param_types,
                   const Type &y_type,
                   const TakeArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  if (args->axis.defined()) {
    const auto* v = args->axis.as<IntValueObj>();
    CHECK(v != nullptr);
    key << v->data;
  }
  return key;
}

MNM_TVMJIT(Take, "mnm.op.take", TakeArgs, TakeNormalizer, TakeTyper, TakeHasher);

Attrs SequenceMaskNormalizer(TVMOpEnv* env, const SequenceMaskArgs* args) {
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(2);
  env->inputs[0] = GetDLTensor(args->x);
  env->inputs[1] = GetDLTensor(args->sequence_length);
  auto attrs = make_object<SequenceMaskAttrs>();
  attrs->mask_value = args->mask_value;
  attrs->axis = args->axis;
  return Attrs(attrs);
}

void SequenceMaskTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  y_type[0] = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0]),
                  GetTensorType(env->inputs[1])};
}

HashKey SequenceMaskHasher(const std::vector<Type>& param_types,
                           const Type &y_type,
                           const SequenceMaskArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->mask_value;
  key << args->axis;
  return key;
}

MNM_TVMJIT(SequenceMask, "mnm.op.sequence_mask", SequenceMaskArgs,
           SequenceMaskNormalizer, SequenceMaskTyper, SequenceMaskHasher);

Attrs BroadcastToNormalizer(TVMOpEnv* env, const BroadcastToArgs* args) {
  using namespace tvm;
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->x);
  auto attrs = make_object<InitOpAttrs>();
  std::vector<IndexExpr> shape;
  shape.reserve(args->shape.size());
  for (size_t i = 0; i < args->shape.size(); ++i) {
    shape.emplace_back(IntImm::make(Int(32), args->shape[i]));
  }
  attrs->shape = Array<relay::IndexExpr>(shape.begin(), shape.end());
  return Attrs(attrs);
}

void BroadcastToTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  y_type[0] = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0])};
}

MNM_TVMJIT(BroadcastTo, "mnm.op.broadcast_to", BroadcastToArgs,
           BroadcastToNormalizer, BroadcastToTyper, GenericHasher);

Attrs TransposeNormalizer(TVMOpEnv* env, const TransposeArgs* args) {
  using namespace tvm;
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->x);
  auto attrs = make_object<TransposeAttrs>();
  std::vector<Integer> axes;
  axes.reserve(args->axes.size());
  for (size_t i = 0; i < args->axes.size(); ++i) {
    axes.emplace_back(args->axes[i]);
  }
  attrs->axes = Array<Integer>(axes.begin(), axes.end());
  return Attrs(attrs);
}

void TransposeTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {GetTensorType(env->inputs[0])};
}

HashKey TransposeHasher(const std::vector<Type>& param_types,
                        const Type& y_type,
                        const TransposeArgs *args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axes;
  return key;
}

MNM_TVMJIT(Transpose, "mnm.op.transpose", TransposeArgs, TransposeNormalizer,
           TransposeTyper, TransposeHasher);

Attrs TransposeDxNormalizer(TVMOpEnv* env, const TransposeDxArgs* args) {
  using namespace tvm;
  CHECK_EQ(env->outputs.size(), 1U);
  env->inputs.resize(3);
  env->inputs[0] = GetDLTensor(args->x);
  env->inputs[1] = GetDLTensor(args->y);
  env->inputs[2] = GetDLTensor(args->dy);
  auto attrs = make_object<TransposeAttrs>();
  std::vector<Integer> axes;
  axes.reserve(args->axes.size());
  for (size_t i = 0; i < args->axes.size(); ++i) {
    axes.emplace_back(args->axes[i]);
  }
  attrs->axes = Array<Integer>(axes.begin(), axes.end());
  return Attrs(attrs);
}

void TransposeDxTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTensorType(env->outputs[0]);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
  };
}

HashKey TransposeDxHasher(const std::vector<Type>& param_types,
                        const Type& y_type,
                        const TransposeDxArgs *args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axes;
  return key;
}

MNM_TVMJIT(TransposeDx, "mnm.op.transpose_dx", TransposeDxArgs, TransposeDxNormalizer,
           TransposeDxTyper, TransposeDxHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
