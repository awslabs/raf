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
class ClipArgs : public ir::AttrsNode<ClipArgs> {
 public:
  value::TensorValue x;
  double a_min;
  double a_max;
  MNM_OP_SCHEMA(ClipArgs, "mnm.args.clip");
};
class ClipDxArgs : public ir::AttrsNode<ClipDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue dy;
  double a_min;
  double a_max;
  MNM_OP_SCHEMA(ClipDxArgs, "mnm.args.clip_dx");
};
class TransposeArgs : public ir::AttrsNode<TransposeArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> axes{};
  MNM_OP_SCHEMA(TransposeArgs, "mnm.args.transpose");
};
class TransposeDxArgs : public ir::AttrsNode<TransposeDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> axes{};
  MNM_OP_SCHEMA(TransposeDxArgs, "mnm.args.transpose_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
