/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/gemm.h
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
class MatmulArgs : public ir::AttrsNode<MatmulArgs> {
 public:
  value::TensorValue a;
  value::TensorValue b;
  bool transpose_a{false};
  bool transpose_b{false};
  MNM_OP_SCHEMA(MatmulArgs, "mnm.args.matmul");
};
class MatmulDabArgs : public ir::AttrsNode<MatmulDabArgs> {
 public:
  value::TensorValue dy;
  value::TensorValue a_or_b;
  bool transpose_dx{false};
  bool transpose_dy{false};
  MNM_OP_SCHEMA(MatmulDabArgs, "mnm.args.matmul_dab");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
