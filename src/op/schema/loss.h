/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/loss.h
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
class LossArgs : public ir::AttrsNode<LossArgs> {
 public:
  value::BaseTensorValue y_true;
  value::BaseTensorValue y_pred;
  MNM_OP_SCHEMA(LossArgs, "mnm.args.loss");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
