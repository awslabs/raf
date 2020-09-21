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
}  // namespace schema
}  // namespace op
}  // namespace mnm
