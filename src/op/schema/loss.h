/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/loss.h
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
class LossArgs : public ir::AttrsNode<LossArgs> {
 public:
  value::TensorValue y_true;
  value::TensorValue y_pred;
  MNM_OP_SCHEMA(LossArgs, "mnm.args.loss");
};
class LossDxArgs : public ir::AttrsNode<LossDxArgs> {
 public:
  value::TensorValue loss;
  value::TensorValue y_true;
  value::TensorValue y_pred;
  MNM_OP_SCHEMA(LossDxArgs, "mnm.args.loss_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
