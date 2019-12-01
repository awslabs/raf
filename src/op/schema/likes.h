/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/likes.h
 * \brief Operator schema. Auto generated. Do not touch.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {
namespace op {
namespace schema {
class CollapseLikeArgs : public ir::AttrsNode<CollapseLikeArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(CollapseLikeArgs, "mnm.args.collapse_like");
};
class ReshapeLikeArgs : public ir::AttrsNode<ReshapeLikeArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(ReshapeLikeArgs, "mnm.args.reshape_like");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
