#pragma once
#include <mnm/op.h>

namespace mnm {
namespace op {
namespace args {

std::vector<int64_t> NormalizeTupleOrInt(const value::Value& value);

class ConvArgs : public ir::AttrsNode<ConvArgs> {
 public:
  value::TensorValue x;
  value::TensorValue w;
  std::vector<int64_t> stride = {1};
  std::vector<int64_t> padding = {0};
  std::vector<int64_t> dilation = {1};
  int64_t groups = 1;

  MNM_OP_SCHEMA(ConvArgs, "mnm.args.conv") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, value::TensorValue, w);
    MNM_ARG_OPTIONAL(2, NormalizeTupleOrInt, stride);
    MNM_ARG_OPTIONAL(3, NormalizeTupleOrInt, padding);
    MNM_ARG_OPTIONAL(4, NormalizeTupleOrInt, dilation);
    MNM_ARG_OPTIONAL(5, int64_t, groups);
  }
};

class PoolArgs : public ir::AttrsNode<PoolArgs> {
 public:
  value::TensorValue x;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride = {};
  std::vector<int64_t> padding = {0};
  std::vector<int64_t> dilation = {1};
  bool ceil_mode = false;
  bool include_pad = true;

  MNM_OP_SCHEMA(PoolArgs, "mnm.args.pool") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, NormalizeTupleOrInt, kernel);
    MNM_ARG_OPTIONAL(2, NormalizeTupleOrInt, stride);
    MNM_ARG_OPTIONAL(3, NormalizeTupleOrInt, padding);
    MNM_ARG_OPTIONAL(4, NormalizeTupleOrInt, dilation);
    MNM_ARG_OPTIONAL(5, bool, ceil_mode);
    MNM_ARG_OPTIONAL(6, bool, include_pad);
  }
};

class SoftmaxArgs : public ir::AttrsNode<SoftmaxArgs> {
 public:
  value::TensorValue x;
  int axis;

  MNM_OP_SCHEMA(SoftmaxArgs, "mnm.args.softmax") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, int, axis);
  }
};

class BatchNormArgs : public ir::AttrsNode<BatchNormArgs> {
 public:
  value::TensorValue x;
  value::TensorValue running_mean;
  value::TensorValue running_var;
  value::TensorValue scale{nullptr};
  value::TensorValue bias{nullptr};
  double eps = 1e-5;
  double momentum = 0.1;

  MNM_OP_SCHEMA(BatchNormArgs, "mnm.args.batch_norm") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, value::TensorValue, running_mean);
    MNM_ARG_REQUIRED(2, value::TensorValue, running_var);
    MNM_ARG_OPTIONAL(3, value::TensorValue, scale);
    MNM_ARG_OPTIONAL(4, value::TensorValue, bias);
    MNM_ARG_OPTIONAL(5, double, eps);
    MNM_ARG_OPTIONAL(6, double, momentum);
  }
};

class LocalResponseNormArgs : public ir::AttrsNode<LocalResponseNormArgs> {
 public:
  value::TensorValue x;
  int64_t size;
  double alpha = 0.0001;
  double beta = 0.75;
  double k = 1.0;

  MNM_OP_SCHEMA(LocalResponseNormArgs, "mnm.args.local_response_norm") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, int64_t, size);
    MNM_ARG_OPTIONAL(2, double, alpha);
    MNM_ARG_OPTIONAL(3, double, beta);
    MNM_ARG_OPTIONAL(4, double, k);
  }
};

}  // namespace args
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace args {

class ConvDxwArgs : public ir::AttrsNode<ConvDxwArgs> {
 public:
  value::TensorValue x_or_w;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups;

  MNM_OP_SCHEMA(ConvDxwArgs, "mnm.args.conv_dxw") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x_or_w);
    MNM_ARG_REQUIRED(1, value::TensorValue, y);
    MNM_ARG_REQUIRED(2, value::TensorValue, dy);
    MNM_ARG_REQUIRED(3, NormalizeTupleOrInt, stride);
    MNM_ARG_REQUIRED(4, NormalizeTupleOrInt, padding);
    MNM_ARG_REQUIRED(5, NormalizeTupleOrInt, dilation);
    MNM_ARG_REQUIRED(6, int64_t, groups);
  }
};

class PoolDxArgs : public ir::AttrsNode<PoolDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride = {};
  std::vector<int64_t> padding = {0};
  std::vector<int64_t> dilation = {1};
  bool ceil_mode = false;
  bool include_pad = true;

  MNM_OP_SCHEMA(PoolDxArgs, "mnm.args.pool_dx") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, value::TensorValue, y);
    MNM_ARG_REQUIRED(2, value::TensorValue, dy);
    MNM_ARG_REQUIRED(3, NormalizeTupleOrInt, kernel);
    MNM_ARG_REQUIRED(4, NormalizeTupleOrInt, stride);
    MNM_ARG_REQUIRED(5, NormalizeTupleOrInt, padding);
    MNM_ARG_REQUIRED(6, NormalizeTupleOrInt, dilation);
    MNM_ARG_REQUIRED(7, bool, ceil_mode);
    MNM_ARG_REQUIRED(8, bool, include_pad);
  }
};

class SoftmaxDxArgs : public ir::AttrsNode<SoftmaxDxArgs> {
 public:
  value::TensorValue x;
  value::TensorValue y;
  value::TensorValue dy;
  int axis;

  MNM_OP_SCHEMA(SoftmaxDxArgs, "mnm.args.softmax_dx") {
    MNM_ARG_REQUIRED(0, value::TensorValue, x);
    MNM_ARG_REQUIRED(1, value::TensorValue, y);
    MNM_ARG_REQUIRED(2, value::TensorValue, dy);
    MNM_ARG_REQUIRED(3, int, axis);
  }
};

}  // namespace args
}  // namespace op
}  // namespace mnm
