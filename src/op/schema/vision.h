/*!
 * Copyright (c) 2020 by Contributors
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
  double score_threshold{0};
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
  double iou_threshold{0.5};
  bool force_suppress{false};
  int64_t top_k{-1};
  int64_t coord_start{2};
  int64_t score_index{1};
  int64_t id_index{0};
  bool return_indices{true};
  bool invalid_to_bottom{false};
  MNM_OP_SCHEMA(NonMaxSuppressionArgs, "mnm.args.non_max_suppression");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
