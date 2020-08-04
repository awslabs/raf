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
}  // namespace schema
}  // namespace op
}  // namespace mnm
