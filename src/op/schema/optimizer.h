/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/optimizer.h
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
class SgdArgs : public ir::AttrsNode<SgdArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dx;
  value::BaseTensorValue v;
  double learning_rate;
  double mu;
  MNM_OP_SCHEMA(SgdArgs, "mnm.args.sgd");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
