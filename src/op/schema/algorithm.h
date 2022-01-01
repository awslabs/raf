/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/algorithm.h
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
class ArgsortArgs : public ir::AttrsNode<ArgsortArgs> {
 public:
  value::BaseTensorValue data;
  int axis{-1};
  bool is_ascend{true};
  std::string dtype{"int"};
  MNM_OP_SCHEMA(ArgsortArgs, "mnm.args.argsort");
};

class SortArgs : public ir::AttrsNode<SortArgs> {
 public:
  value::BaseTensorValue data;
  int axis{-1};
  bool is_ascend{true};
  MNM_OP_SCHEMA(SortArgs, "mnm.args.sort");
};

class TopkArgs : public ir::AttrsNode<TopkArgs> {
 public:
  value::BaseTensorValue data;
  value::Value k;
  int axis{-1};
  std::string ret_type{"both"};
  bool is_ascend{false};
  std::string dtype{"int64_t"};
  MNM_OP_SCHEMA(TopkArgs, "mnm.args.topk");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
