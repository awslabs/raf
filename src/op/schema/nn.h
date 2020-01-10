/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/nn.h
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
class BatchNormArgs : public ir::AttrsNode<BatchNormArgs> {
 public:
  value::TensorValue x;
  value::TensorValue running_mean;
  value::TensorValue running_var;
  value::TensorValue w{nullptr};
  value::TensorValue b{nullptr};
  double momentum{0.1};
  double eps{1e-05};
  MNM_OP_SCHEMA(BatchNormArgs, "mnm.args.batch_norm");
};
class BatchNormTrainDxwbArgs : public ir::AttrsNode<BatchNormTrainDxwbArgs> {
 public:
  value::TensorValue dy;
  value::TensorValue x;
  value::TensorValue w;
  value::TensorValue b;
  double eps;
  MNM_OP_SCHEMA(BatchNormTrainDxwbArgs, "mnm.args.batch_norm_train_dxwb");
};
class ConvArgs : public ir::AttrsNode<ConvArgs> {
 public:
  value::TensorValue x;
  value::TensorValue w;
  std::vector<int64_t> stride{1};
  std::vector<int64_t> padding{0};
  std::vector<int64_t> dilation{1};
  int64_t groups{1};
  MNM_OP_SCHEMA(ConvArgs, "mnm.args.conv");
};
class ConvDxwArgs : public ir::AttrsNode<ConvDxwArgs> {
 public:
  value::TensorValue x_or_w;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups;
  MNM_OP_SCHEMA(ConvDxwArgs, "mnm.args.conv_dxw");
};
class LocalResponseNormArgs : public ir::AttrsNode<LocalResponseNormArgs> {
 public:
  value::TensorValue x;
  int64_t size;
  double alpha{0.0001};
  double beta{0.75};
  double k{1.0};
  MNM_OP_SCHEMA(LocalResponseNormArgs, "mnm.args.local_response_norm");
};
class PoolArgs : public ir::AttrsNode<PoolArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding{0};
  std::vector<int64_t> dilation{1};
  bool ceil_mode{false};
  bool include_pad{true};
  MNM_OP_SCHEMA(PoolArgs, "mnm.args.pool");
};
class PoolDxArgs : public ir::AttrsNode<PoolDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  bool include_pad;
  MNM_OP_SCHEMA(PoolDxArgs, "mnm.args.pool_dx");
};
class SoftmaxArgs : public ir::AttrsNode<SoftmaxArgs> {
 public:
  value::TensorValue x;
  int axis{-1};
  MNM_OP_SCHEMA(SoftmaxArgs, "mnm.args.softmax");
};
class SoftmaxDxArgs : public ir::AttrsNode<SoftmaxDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  int axis{-1};
  MNM_OP_SCHEMA(SoftmaxDxArgs, "mnm.args.softmax_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
