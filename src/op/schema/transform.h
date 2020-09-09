/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/transform.h
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
class CastArgs : public ir::AttrsNode<CastArgs> {
 public:
  value::BaseTensorValue data;
  std::string dtype;
  MNM_OP_SCHEMA(CastArgs, "mnm.args.cast");
};

class CastLikeArgs : public ir::AttrsNode<CastLikeArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue dtype_like;
  MNM_OP_SCHEMA(CastLikeArgs, "mnm.args.cast_like");
};

class ClipArgs : public ir::AttrsNode<ClipArgs> {
 public:
  value::BaseTensorValue x;
  double a_min;
  double a_max;
  MNM_OP_SCHEMA(ClipArgs, "mnm.args.clip");
};

class ClipDxArgs : public ir::AttrsNode<ClipDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  double a_min;
  double a_max;
  MNM_OP_SCHEMA(ClipDxArgs, "mnm.args.clip_dx");
};

class GatherNdArgs : public ir::AttrsNode<GatherNdArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue indices;
  MNM_OP_SCHEMA(GatherNdArgs, "mnm.args.gather_nd");
};

class GatherNdDxArgs : public ir::AttrsNode<GatherNdDxArgs> {
 public:
  value::BaseTensorValue data;
  value::BaseTensorValue indices;
  value::BaseTensorValue dy;
  MNM_OP_SCHEMA(GatherNdDxArgs, "mnm.args.gather_nd_dx");
};

class ReverseArgs : public ir::AttrsNode<ReverseArgs> {
 public:
  value::BaseTensorValue x;
  int axis{0};
  MNM_OP_SCHEMA(ReverseArgs, "mnm.args.reverse");
};

class TransposeArgs : public ir::AttrsNode<TransposeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axes{};
  MNM_OP_SCHEMA(TransposeArgs, "mnm.args.transpose");
};

class TransposeDxArgs : public ir::AttrsNode<TransposeDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  std::vector<int64_t> axes{};
  MNM_OP_SCHEMA(TransposeDxArgs, "mnm.args.transpose_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
