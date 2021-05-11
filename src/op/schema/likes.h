/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/likes.h
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
class CollapseLikeArgs : public ir::AttrsNode<CollapseLikeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(CollapseLikeArgs, "mnm.args.collapse_like");
};

class ReshapeArgs : public ir::AttrsNode<ReshapeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> shape;
  bool reverse{false};
  MNM_OP_SCHEMA(ReshapeArgs, "mnm.args.reshape");
};

class ResizeArgs : public ir::AttrsNode<ResizeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> size;
  std::string layout{"NCHW"};
  std::string method{"bilinear"};
  std::string coordinate_transformation_mode{"half_pixel"};
  std::string rounding_method{};
  float bicubic_alpha{-0.5};
  int bicubic_exclude{0};
  std::string out_dtype{};
  MNM_OP_SCHEMA(ResizeArgs, "mnm.args.resize");
};

class SumArgs : public ir::AttrsNode<SumArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axis{};
  std::vector<int64_t> keepdims{0};
  bool exclude{false};
  MNM_OP_SCHEMA(SumArgs, "mnm.args.sum");
};

class SumDxArgs : public ir::AttrsNode<SumDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  std::vector<int64_t> keepdims{0};
  bool exclude{false};
  MNM_OP_SCHEMA(SumDxArgs, "mnm.args.sum_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
