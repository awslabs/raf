/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/init.h
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
class ShapeDtypeArgs : public ir::AttrsNode<ShapeDtypeArgs> {
 public:
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(ShapeDtypeArgs, "mnm.args.shape_dtype");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
