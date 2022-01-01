/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/optimizer.h
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
class LansArgs : public ir::AttrsNode<LansArgs> {
 public:
  std::vector<value::BaseTensorValue> tensor_list;
  value::BaseTensorValue step;
  float learning_rate;
  float beta1;
  float beta2;
  float eps;
  int bias_correction;
  float weight_decay;
  int grad_averaging;
  int mode;
  bool normalize_grad;
  MNM_OP_SCHEMA(LansArgs, "mnm.args.lans");
};

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
