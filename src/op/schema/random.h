/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/random.h
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
class ThreefryGenerateArgs : public ir::AttrsNode<ThreefryGenerateArgs> {
 public:
  value::BaseTensorValue key;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(ThreefryGenerateArgs, "mnm.args.threefry_generate");
};

class ThreefrySplitArgs : public ir::AttrsNode<ThreefrySplitArgs> {
 public:
  value::BaseTensorValue key;
  MNM_OP_SCHEMA(ThreefrySplitArgs, "mnm.args.threefry_split");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
