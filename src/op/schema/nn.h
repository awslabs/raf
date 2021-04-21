/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/nn.h
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
class AdaptivePoolArgs : public ir::AttrsNode<AdaptivePoolArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> shape;
  std::string layout{"NCHW"};
  MNM_OP_SCHEMA(AdaptivePoolArgs, "mnm.args.adaptive_pool");
};

class AdaptivePoolDxArgs : public ir::AttrsNode<AdaptivePoolDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(AdaptivePoolDxArgs, "mnm.args.adaptive_pool_dx");
};

class BatchNormArgs : public ir::AttrsNode<BatchNormArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue running_mean;
  value::BaseTensorValue running_var;
  value::BaseTensorValue w{nullptr};
  value::BaseTensorValue b{nullptr};
  double momentum{0.1};
  double eps{1e-05};
  MNM_OP_SCHEMA(BatchNormArgs, "mnm.args.batch_norm");
};

class BatchNormTrainDxwbArgs : public ir::AttrsNode<BatchNormTrainDxwbArgs> {
 public:
  value::BaseTensorValue dy;
  value::BaseTensorValue x;
  value::BaseTensorValue w;
  value::BaseTensorValue b;
  double eps;
  MNM_OP_SCHEMA(BatchNormTrainDxwbArgs, "mnm.args.batch_norm_train_dxwb");
};

class BiasAddArgs : public ir::AttrsNode<BiasAddArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue bias;
  int64_t axis{1};
  MNM_OP_SCHEMA(BiasAddArgs, "mnm.args.bias_add");
};

class BroadcastToArgs : public ir::AttrsNode<BroadcastToArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(BroadcastToArgs, "mnm.args.broadcast_to");
};

class BroadcastToLikeArgs : public ir::AttrsNode<BroadcastToLikeArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue broadcast_type;
  MNM_OP_SCHEMA(BroadcastToLikeArgs, "mnm.args.broadcast_to_like");
};

class ConcatenateArgs : public ir::AttrsNode<ConcatenateArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  int axis{0};
  MNM_OP_SCHEMA(ConcatenateArgs, "mnm.args.concatenate");
};

class ConvArgs : public ir::AttrsNode<ConvArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue w;
  std::vector<int64_t> stride{1};
  std::vector<int64_t> padding{0};
  std::vector<int64_t> dilation{1};
  int64_t groups{1};
  std::string layout{"NCHW"};
  std::string kernel_layout{"OIHW"};
  std::string out_layout{"NCHW"};
  MNM_OP_SCHEMA(ConvArgs, "mnm.args.conv");
};

class ConvDxwArgs : public ir::AttrsNode<ConvDxwArgs> {
 public:
  value::BaseTensorValue x_or_w;
  ir::Optional<value::BaseTensorValue> y;
  value::BaseTensorValue dy;
  ir::Optional<ir::Array<value::IntValue>> shape;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups;
  MNM_OP_SCHEMA(ConvDxwArgs, "mnm.args.conv_dxw");
};

class DropoutArgs : public ir::AttrsNode<DropoutArgs> {
 public:
  value::BaseTensorValue x;
  double p{0.5};
  ir::Optional<value::BaseTensorValue> in_states{nullptr};
  MNM_OP_SCHEMA(DropoutArgs, "mnm.args.dropout");
};

class LayerNormArgs : public ir::AttrsNode<LayerNormArgs> {
 public:
  value::BaseTensorValue x;
  ir::Optional<value::BaseTensorValue> scale{nullptr};
  ir::Optional<value::BaseTensorValue> bias{nullptr};
  int64_t axis{-1};
  double eps{1e-05};
  MNM_OP_SCHEMA(LayerNormArgs, "mnm.args.layer_norm");
};

class LayerNormDxArgs : public ir::AttrsNode<LayerNormDxArgs> {
 public:
  value::BaseTensorValue x;
  ir::Optional<value::BaseTensorValue> scale;
  value::BaseTensorValue dy;
  int64_t axis{-1};
  double eps{1e-05};
  MNM_OP_SCHEMA(LayerNormDxArgs, "mnm.args.layer_norm_dx");
};

class LocalResponseNormArgs : public ir::AttrsNode<LocalResponseNormArgs> {
 public:
  value::BaseTensorValue x;
  int64_t size;
  double alpha{0.0001};
  double beta{0.75};
  double k{1.0};
  MNM_OP_SCHEMA(LocalResponseNormArgs, "mnm.args.local_response_norm");
};

class MeshGridArgs : public ir::AttrsNode<MeshGridArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  MNM_OP_SCHEMA(MeshGridArgs, "mnm.args.mesh_grid");
};

class PadArgs : public ir::AttrsNode<PadArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> pad_width;
  double pad_value{0.0};
  std::string pad_mode{"constant"};
  MNM_OP_SCHEMA(PadArgs, "mnm.args.pad");
};

class PoolArgs : public ir::AttrsNode<PoolArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding{0};
  std::vector<int64_t> dilation{1};
  bool ceil_mode{false};
  bool include_pad{true};
  std::string layout{"NCHW"};
  MNM_OP_SCHEMA(PoolArgs, "mnm.args.pool");
};

class PoolDxArgs : public ir::AttrsNode<PoolDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  std::vector<int64_t> kernel;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  bool include_pad;
  MNM_OP_SCHEMA(PoolDxArgs, "mnm.args.pool_dx");
};

class RepeatArgs : public ir::AttrsNode<RepeatArgs> {
 public:
  value::BaseTensorValue x;
  int repeats;
  value::Value axis{nullptr};
  MNM_OP_SCHEMA(RepeatArgs, "mnm.args.repeat");
};

class RepeatDxArgs : public ir::AttrsNode<RepeatDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  int repeats;
  value::Value axis{nullptr};
  MNM_OP_SCHEMA(RepeatDxArgs, "mnm.args.repeat_dx");
};

class ReverseSequenceArgs : public ir::AttrsNode<ReverseSequenceArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue sequence_length;
  int seq_axis{1};
  int batch_axis{0};
  MNM_OP_SCHEMA(ReverseSequenceArgs, "mnm.args.reverse_sequence");
};

class SequenceMaskArgs : public ir::AttrsNode<SequenceMaskArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue sequence_length;
  double mask_value{0.0};
  int axis{0};
  MNM_OP_SCHEMA(SequenceMaskArgs, "mnm.args.sequence_mask");
};

class SoftmaxArgs : public ir::AttrsNode<SoftmaxArgs> {
 public:
  value::BaseTensorValue x;
  int axis{-1};
  MNM_OP_SCHEMA(SoftmaxArgs, "mnm.args.softmax");
};

class SoftmaxDxArgs : public ir::AttrsNode<SoftmaxDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  int axis{-1};
  MNM_OP_SCHEMA(SoftmaxDxArgs, "mnm.args.softmax_dx");
};

class SplitArgs : public ir::AttrsNode<SplitArgs> {
 public:
  value::BaseTensorValue x;
  value::Value indices_or_sections{nullptr};
  int axis{0};
  MNM_OP_SCHEMA(SplitArgs, "mnm.args.split");
};

class SqueezeArgs : public ir::AttrsNode<SqueezeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axis{};
  MNM_OP_SCHEMA(SqueezeArgs, "mnm.args.squeeze");
};

class StackArgs : public ir::AttrsNode<StackArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  int axis{0};
  MNM_OP_SCHEMA(StackArgs, "mnm.args.stack");
};

class TakeArgs : public ir::AttrsNode<TakeArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue indices;
  value::Value axis{nullptr};
  std::string mode{"clip"};
  MNM_OP_SCHEMA(TakeArgs, "mnm.args.take");
};

class TakeDxArgs : public ir::AttrsNode<TakeDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  value::BaseTensorValue indices;
  value::Value axis{nullptr};
  std::string mode{"clip"};
  MNM_OP_SCHEMA(TakeDxArgs, "mnm.args.take_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
