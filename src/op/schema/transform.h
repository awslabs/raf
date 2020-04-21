/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/transform.h
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
class TransposeArgs : public ir::AttrsNode<TransposeArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> axes;
  MNM_OP_SCHEMA(TransposeArgs, "mnm.args.transpose");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
