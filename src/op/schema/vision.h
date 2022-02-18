/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Auto generated. Do not touch.
 * \file src/op/schema/vision.h
 * \brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {
namespace op {
namespace schema {
class GetValidCountsArgs : public ir::AttrsNode<GetValidCountsArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue score_threshold;
  int64_t id_index{0};
  int64_t score_index{1};
  MNM_OP_SCHEMA(GetValidCountsArgs, "mnm.args.get_valid_counts");
};

class NonMaxSuppressionArgs : public ir::AttrsNode<NonMaxSuppressionArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue valid_count;
  value::BaseTensorValue indices;
  value::BaseTensorValue max_output_size;
  value::BaseTensorValue iou_threshold;
  bool force_suppress{false};
  int64_t top_k{-1};
  int64_t coord_start{2};
  int64_t score_index{1};
  int64_t id_index{0};
  bool return_indices{true};
  bool invalid_to_bottom{false};
  MNM_OP_SCHEMA(NonMaxSuppressionArgs, "mnm.args.non_max_suppression");
};

class RoiAlignArgs : public ir::AttrsNode<RoiAlignArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue rois;
  std::vector<int64_t> pooled_size;
  double spatial_scale;
  int64_t sample_ratio{-1};
  std::string layout{"NCHW"};
  std::string mode{"avg"};
  MNM_OP_SCHEMA(RoiAlignArgs, "mnm.args.roi_align");
};

class RoiAlignDxArgs : public ir::AttrsNode<RoiAlignDxArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue rois;
  value::BaseTensorValue dy;
  std::vector<int64_t> pooled_size;
  double spatial_scale;
  int64_t sample_ratio{-1};
  std::string layout{"NCHW"};
  std::string mode{"avg"};
  MNM_OP_SCHEMA(RoiAlignDxArgs, "mnm.args.roi_align_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
