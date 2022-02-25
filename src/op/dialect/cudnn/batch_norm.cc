/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/batch_norm.cc
 * \brief CUDNN BatchNorm operators.
 */
#include "../../schema/nn.h"
#include "./cudnn_utils.h"
#include "raf/ir.h"
#include "raf/op_utils.h"

namespace raf {
namespace op {
namespace cudnn {

using namespace raf::value;
using namespace raf::ir;

static auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");

class BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  cudnnTensorDescriptor_t yDesc;
  double epsilon;
  double exponentialAverageFactor;

  explicit BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference(
      const CallValues& cv) {
    auto op = Op::Get("raf.op.batch_norm_infer");
    this->arg_indices = {
        fschema_index[op]("x"),           fschema_index[op]("running_mean"),
        fschema_index[op]("running_var"), fschema_index[op]("w"),
        fschema_index[op]("b"),
    };
    auto args = cv->args.as<raf::op::schema::BatchNormArgs>();
    DLTensor* x = args->x;
    DLTensor* w = args->w;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, 1, 2, 3, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    epsilon = args->eps;
    exponentialAverageFactor = args->momentum;
  }

 public:
  ~BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.batch_norm_infer"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::BatchNormArgs>();
    DLTensor* x = args->x;
    DLTensor* w = args->w;
    DLTensor* out = cv->out;
    DLTensor* running_mean = args->running_mean;
    DLTensor* running_var = args->running_var;
    DLTensor* b = args->b;
    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out->data, bnScaleBiasMeanVarDesc, w->data, b->data, running_mean->data,
        running_var->data, epsilon));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 5);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* running_mean = Downcast<TensorValue>(inputs[1]);
    DLTensor* running_var = Downcast<TensorValue>(inputs[2]);
    DLTensor* w = Downcast<TensorValue>(inputs[3]);
    DLTensor* b = Downcast<TensorValue>(inputs[4]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out->data, bnScaleBiasMeanVarDesc, w->data, b->data, running_mean->data,
        running_var->data, epsilon));
  }

  static OpEnv* make(const CallValues& cv) {
    return new BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, batch_norm_infer, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.batch_norm_infer",
                 BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference::make);

class BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  cudnnTensorDescriptor_t yDesc;
  double epsilon;
  double exponentialAverageFactor;

  explicit BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining(const CallValues& cv) {
    auto op = Op::Get("raf.op.batch_norm_train");
    this->arg_indices = {
        fschema_index[op]("x"),           fschema_index[op]("running_mean"),
        fschema_index[op]("running_var"), fschema_index[op]("w"),
        fschema_index[op]("b"),
    };
    auto args = cv->args.as<raf::op::schema::BatchNormArgs>();
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* w = args->w;
    DLTensor* out0 = tv->fields[0];
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
    auto yDesc_tt = SquashTensorShape(out0, {0, 1, 2, 3, out0->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    epsilon = args->eps;
    exponentialAverageFactor = args->momentum;
  }

 public:
  ~BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.batch_norm_train"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::BatchNormArgs>();
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* w = args->w;
    DLTensor* out0 = tv->fields[0];
    DLTensor* running_mean = args->running_mean;
    DLTensor* running_var = args->running_var;
    DLTensor* b = args->b;
    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out0->data, bnScaleBiasMeanVarDesc, w->data, b->data, exponentialAverageFactor,
        running_mean->data, running_var->data, epsilon, nullptr, nullptr));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 5);
    TupleValue tv = Downcast<TupleValue>(output);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* running_mean = Downcast<TensorValue>(inputs[1]);
    DLTensor* running_var = Downcast<TensorValue>(inputs[2]);
    DLTensor* w = Downcast<TensorValue>(inputs[3]);
    DLTensor* b = Downcast<TensorValue>(inputs[4]);
    DLTensor* out0 = Downcast<TensorValue>(tv->fields[0]);
    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out0->data, bnScaleBiasMeanVarDesc, w->data, b->data, exponentialAverageFactor,
        running_mean->data, running_var->data, epsilon, nullptr, nullptr));
  }

  static OpEnv* make(const CallValues& cv) {
    return new BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, batch_norm_train, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.batch_norm_train",
                 BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining::make);

class BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnTensorDescriptor_t dBnScaleBiasDesc;
  double epsilon;

  explicit BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward(const CallValues& cv) {
    auto op = Op::Get("raf.op.batch_norm_train_dxwb");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::BatchNormTrainDxwbArgs>();
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* dy = args->dy;
    DLTensor* out0 = tv->fields[0];
    DLTensor* out1 = tv->fields[1];
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, 1, 2, 3, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out0, {0, 1, 2, 3, out0->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    auto dBnScaleBiasDesc_tt = SquashTensorShape(out1, {0, 0, 1, out1->ndim});
    dBnScaleBiasDesc = NormalizeTensorType(dBnScaleBiasDesc_tt);
    epsilon = args->eps;
  }

 public:
  ~BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dBnScaleBiasDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.batch_norm_train_dxwb"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::BatchNormTrainDxwbArgs>();
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* dy = args->dy;
    DLTensor* out0 = tv->fields[0];
    DLTensor* out1 = tv->fields[1];
    DLTensor* w = args->w;
    CUDNN_CALL(cudnnBatchNormalizationBackward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(out0->dtype).const_addr<1>(), CUDNNDType(out0->dtype).const_addr<0>(),
        CUDNNDType(out1->dtype).const_addr<1>(), CUDNNDType(out1->dtype).const_addr<0>(), xDesc,
        x->data, dyDesc, dy->data, dxDesc, out0->data, dBnScaleBiasDesc, w->data, out1->data,
        tv->fields[2].operator DLTensor*()->data, epsilon, nullptr, nullptr));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    TupleValue tv = Downcast<TupleValue>(output);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* w = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out0 = Downcast<TensorValue>(tv->fields[0]);
    DLTensor* out1 = Downcast<TensorValue>(tv->fields[1]);
    CUDNN_CALL(cudnnBatchNormalizationBackward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(out0->dtype).const_addr<1>(), CUDNNDType(out0->dtype).const_addr<0>(),
        CUDNNDType(out1->dtype).const_addr<1>(), CUDNNDType(out1->dtype).const_addr<0>(), xDesc,
        x->data, dyDesc, dy->data, dxDesc, out0->data, dBnScaleBiasDesc, w->data, out1->data,
        tv->fields[2].operator DLTensor*()->data, epsilon, nullptr, nullptr));
  }

  static OpEnv* make(const CallValues& cv) {
    return new BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, batch_norm_train_dxwb, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.batch_norm_train_dxwb",
                 BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward::make);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
