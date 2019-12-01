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
}  // namespace schema
}  // namespace op
}  // namespace mnm
