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

std::vector<Value> GetValidCountSchema2Args(const GetValidCountsArgs* args) {
  return {args->data, args->score_threshold};
}

std::vector<std::string> GetValidCountSchemaArgNames(const op::CallValues& call) {
  return {"data", "score_threshold"};
}

Attrs GetValidCountsSchema2Attrs(const GetValidCountsArgs* args) {
  auto attrs = make_object<GetValidCountsAttrs>();
  attrs->score_threshold =
      FloatImm(DataType::Float(32), GetScalarValueData<float>(args->score_threshold));
  attrs->id_index = args->id_index;
  attrs->score_index = args->score_index;
  return Attrs(attrs);
}

HashKey GetValidCountsHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const GetValidCountsArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->score_threshold;
  key << args->id_index;
  key << args->score_index;
  return key;
}

MNM_TVMJIT(GetValidCounts, "mnm.op.get_valid_counts", GetValidCountsArgs, GetValidCountSchema2Args,
           GetValidCountSchemaArgNames, GetValidCountsSchema2Attrs, GetValidCountsHasher);

std::vector<Value> NonMaxSuppressionSchema2Args(const NonMaxSuppressionArgs* args) {
  return {args->data, args->valid_count, args->indices, args->max_output_size, args->iou_threshold};
}

std::vector<std::string> NonMaxSuppressionSchemaArgNames(const op::CallValues& call) {
  return {"data", "valid_count", "indices", "max_output_size", "iou_threshold"};
}

Attrs NonMaxSuppressionSchema2Attrs(const NonMaxSuppressionArgs* args) {
  auto attrs = make_object<NonMaximumSuppressionAttrs>();
  attrs->force_suppress = args->force_suppress;
  attrs->top_k = args->top_k;
  attrs->coord_start = args->coord_start;
  attrs->score_index = args->score_index;
  attrs->id_index = args->id_index;
  attrs->return_indices = args->return_indices;
  attrs->invalid_to_bottom = args->invalid_to_bottom;
  return Attrs(attrs);
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
           NonMaxSuppressionSchema2Args, NonMaxSuppressionSchemaArgNames,
           NonMaxSuppressionSchema2Attrs, NonMaxSuppressionHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
