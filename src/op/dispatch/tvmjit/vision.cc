/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/vision.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/vision.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/vision.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

Attrs GetValidCountsNormalizer(TVMOpEnv* env, const GetValidCountsArgs* args) {
  CHECK_EQ(env->outputs.size(), 3U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->data);
  auto attrs = make_object<GetValidCountsAttrs>();
  attrs->score_threshold = args->score_threshold;
  attrs->id_index = args->id_index;
  attrs->score_index = args->score_index;
  return Attrs(attrs);
}

void GetValidCountsTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTupleType(env->outputs);
  *param_types = {GetTensorType(env->inputs[0])};
}

HashKey GetValidCountsHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const GetValidCountsArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->score_threshold;
  key << args->id_index;
  key << args->score_index;
  return key;
}

MNM_TVMJIT(GetValidCounts, "mnm.op.get_valid_counts", GetValidCountsArgs, GetValidCountsNormalizer,
           GetValidCountsTyper, GetValidCountsHasher);

Attrs NonMaxSuppressionNormalizer(TVMOpEnv* env, const NonMaxSuppressionArgs* args) {
  env->inputs.resize(4);
  env->inputs[0] = GetDLTensor(args->data);
  env->inputs[1] = GetDLTensor(args->valid_count);
  env->inputs[2] = GetDLTensor(args->indices);
  env->inputs[3] = GetDLTensor(args->max_output_size);
  auto attrs = make_object<NonMaximumSuppressionAttrs>();
  attrs->iou_threshold = args->iou_threshold;
  attrs->force_suppress = args->force_suppress;
  attrs->top_k = args->top_k;
  attrs->coord_start = args->coord_start;
  attrs->score_index = args->score_index;
  attrs->id_index = args->id_index;
  attrs->return_indices = args->return_indices;
  attrs->invalid_to_bottom = args->invalid_to_bottom;
  return Attrs(attrs);
}

void NonMaxSuppressionTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTupleType(env->outputs);
  *param_types = {
      GetTensorType(env->inputs[0]),
      GetTensorType(env->inputs[1]),
      GetTensorType(env->inputs[2]),
      GetTensorType(env->inputs[3]),
  };
}

HashKey NonMaxSuppressionHasher(const std::vector<Type>& param_types, const Type& y_type,
                                const NonMaxSuppressionArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->iou_threshold;
  key << args->force_suppress;
  key << args->top_k;
  key << args->coord_start;
  key << args->score_index;
  key << args->id_index;
  key << args->return_indices;
  key << args->invalid_to_bottom;
  return key;
}

MNM_TVMJIT(NonMaxSuppression, "mnm.op.non_max_suppression", NonMaxSuppressionArgs,
           NonMaxSuppressionNormalizer, NonMaxSuppressionTyper, NonMaxSuppressionHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
