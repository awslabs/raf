/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/vision.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/vision.h>
#include <raf/value.h>
#include <array>
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/vision.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
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

RAF_TVM(get_valid_counts, GetValidCounts, GetValidCountsArgs, GetValidCountSchema2Args,
        GetValidCountSchemaArgNames, GetValidCountsSchema2Attrs, GetValidCountsHasher, kOpaque);

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

RAF_TVM(non_max_suppression, NonMaxSuppression, NonMaxSuppressionArgs, NonMaxSuppressionSchema2Args,
        NonMaxSuppressionSchemaArgNames, NonMaxSuppressionSchema2Attrs, NonMaxSuppressionHasher,
        kOpaque);

std::vector<Value> RoiAlignSchema2Args(const RoiAlignArgs* args) {
  return {args->data, args->rois};
}

std::vector<std::string> RoiAlignSchemaArgNames(const op::CallValues& call) {
  return {"data", "rois"};
}

Attrs RoiAlignSchema2Attrs(const RoiAlignArgs* args) {
  auto attrs = make_object<ROIAlignAttrs>();
  std::vector<int64_t> pooled_size = args->pooled_size;
  for (int i = 0; i < pooled_size.size(); ++i) {
    attrs->pooled_size.push_back(IntImm(tvm::runtime::DataType::Int(64), pooled_size[i]));
  }
  attrs->spatial_scale = args->spatial_scale;
  attrs->sample_ratio = args->sample_ratio;
  attrs->layout = args->layout;
  attrs->mode = args->mode;
  return Attrs(attrs);
}

HashKey RoiAlignHasher(const std::vector<Type>& param_types, const Type& y_type,
                       const RoiAlignArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->pooled_size;
  key << args->spatial_scale;
  key << args->sample_ratio;
  key << args->layout;
  key << args->mode;
  return key;
}

RAF_TVM(roi_align, RoiAlign, RoiAlignArgs, RoiAlignSchema2Args, RoiAlignSchemaArgNames,
        RoiAlignSchema2Attrs, RoiAlignHasher, kOutEWiseFusable);

std::vector<Value> RoiAlignDxSchema2Args(const RoiAlignDxArgs* args) {
  return {args->data, args->rois, args->dy};
}

std::vector<std::string> RoiAlignDxSchemaArgNames(const op::CallValues& call) {
  return {"data", "rois", "dy"};
}

Attrs RoiAlignDxSchema2Attrs(const RoiAlignDxArgs* args) {
  auto attrs = make_object<ROIAlignAttrs>();
  std::vector<int64_t> pooled_size = args->pooled_size;
  for (int i = 0; i < pooled_size.size(); ++i) {
    attrs->pooled_size.push_back(IntImm(tvm::runtime::DataType::Int(64), pooled_size[i]));
  }
  attrs->spatial_scale = args->spatial_scale;
  attrs->sample_ratio = args->sample_ratio;
  attrs->layout = args->layout;
  attrs->mode = args->mode;
  return Attrs(attrs);
}

HashKey RoiAlignDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                         const RoiAlignDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->pooled_size;
  key << args->spatial_scale;
  key << args->sample_ratio;
  key << args->layout;
  key << args->mode;
  return key;
}

RAF_TVM(roi_align_dx, RoiAlignDx, RoiAlignDxArgs, RoiAlignDxSchema2Args, RoiAlignDxSchemaArgNames,
        RoiAlignDxSchema2Attrs, RoiAlignDxHasher, kOutEWiseFusable);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
