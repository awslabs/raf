/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/reduce.h
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
class L2NormArgs : public ir::AttrsNode<L2NormArgs> {
 public:
  value::BaseTensorValue x;
  MNM_OP_SCHEMA(L2NormArgs, "mnm.args.l2norm");
};

class MeanDxArgs : public ir::AttrsNode<MeanDxArgs> {
 public:
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  std::vector<int64_t> x_shape{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(MeanDxArgs, "mnm.args.mean_dx");
};

class ProdDxArgs : public ir::AttrsNode<ProdDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(ProdDxArgs, "mnm.args.prod_dx");
};

class ReduceArgs : public ir::AttrsNode<ReduceArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(ReduceArgs, "mnm.args.reduce");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
