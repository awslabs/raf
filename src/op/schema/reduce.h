/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/reduce.h
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
class ReduceArgs : public ir::AttrsNode<ReduceArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  MNM_OP_SCHEMA(ReduceArgs, "mnm.args.reduce");
};
class ReduceDxArgs : public ir::AttrsNode<ReduceDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  MNM_OP_SCHEMA(ReduceDxArgs, "mnm.args.reduce_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
