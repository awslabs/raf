/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/init.h
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
class InitOpArgs : public ir::AttrsNode<InitOpArgs> {
 public:
  std::vector<int64_t> shape;
  std::string dtype{"int"};
  std::string device{"cpu"};
  MNM_OP_SCHEMA(InitOpArgs, "mnm.args.init_op");
};

class OneHotArgs : public ir::AttrsNode<OneHotArgs> {
 public:
  value::BaseTensorValue indices;
  value::BaseTensorValue on_value;
  value::BaseTensorValue off_value;
  int64_t depth;
  int64_t axis{-1};
  std::string dtype{"int"};
  std::string device{"cpu"};
  MNM_OP_SCHEMA(OneHotArgs, "mnm.args.one_hot");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
