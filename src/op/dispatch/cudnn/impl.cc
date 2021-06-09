/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/dispatch/cudnn/impl.cc
 * \brief Operator schema.
 */
#include "../../schema/algorithm.h"
#include "../../schema/annotation.h"
#include "../../schema/communication.h"
#include "../../schema/init.h"
#include "../../schema/likes.h"
#include "../../schema/list_args.h"
#include "../../schema/loss.h"
#include "../../schema/memory.h"
#include "../../schema/nn.h"
#include "../../schema/optimizer.h"
#include "../../schema/random.h"
#include "../../schema/reduce.h"
#include "../../schema/stream.h"
#include "../../schema/transform.h"
#include "../../schema/ufunc.h"
#include "../../schema/vision.h"
#include "../../schema/vm.h"
#include "./cudnn_utils.h"
#include "mnm/ir.h"
#include "mnm/op_utils.h"

namespace mnm {
namespace op {
namespace cudnn {
namespace generated {

using namespace mnm::value;
using namespace mnm::ir;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;
using common::shape_utils::PadDims;
using common::shape_utils::Shape2Strides;
using dmlc::BeginPtr;

static auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");

MetaCache<cudnnConvolutionBwdDataAlgoPerf_t> CacheForcudnnConvolutionBwdDataAlgoPerf_t;
cudnnConvolutionBwdDataAlgoPerf_t FindcudnnConvolutionBwdDataAlgoPerf_tWrapper(
    const std::vector<uint8_t>& key, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc) {
  if (auto* val = CacheForcudnnConvolutionBwdDataAlgoPerf_t.Get(key)) {
    return *val;
  }
  int cnt;
  cudnnConvolutionBwdDataAlgoPerf_t res;
  CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
      CUDNNThreadEntry::ThreadLocal()->handle, wDesc, dyDesc, convDesc, dxDesc, 1, &cnt, &res));
  if (res.status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm " << cudnnGetErrorString(res.status);
    throw;
  }
  CacheForcudnnConvolutionBwdDataAlgoPerf_t.Set(key, res);
  return res;
}

MetaCache<cudnnConvolutionBwdFilterAlgoPerf_t> CacheForcudnnConvolutionBwdFilterAlgoPerf_t;
cudnnConvolutionBwdFilterAlgoPerf_t FindcudnnConvolutionBwdFilterAlgoPerf_tWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc) {
  if (auto* val = CacheForcudnnConvolutionBwdFilterAlgoPerf_t.Get(key)) {
    return *val;
  }
  int cnt;
  cudnnConvolutionBwdFilterAlgoPerf_t res;
  CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
      CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, 1, &cnt, &res));
  if (res.status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm " << cudnnGetErrorString(res.status);
    throw;
  }
  CacheForcudnnConvolutionBwdFilterAlgoPerf_t.Set(key, res);
  return res;
}

MetaCache<cudnnConvolutionFwdAlgoPerf_t> CacheForcudnnConvolutionFwdAlgoPerf_t;
cudnnConvolutionFwdAlgoPerf_t FindcudnnConvolutionFwdAlgoPerf_tWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc) {
  if (auto* val = CacheForcudnnConvolutionFwdAlgoPerf_t.Get(key)) {
    return *val;
  }
  int cnt;
  cudnnConvolutionFwdAlgoPerf_t res;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(CUDNNThreadEntry::ThreadLocal()->handle, xDesc,
                                                  wDesc, convDesc, yDesc, 1, &cnt, &res));
  if (res.status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm " << cudnnGetErrorString(res.status);
    throw;
  }
  CacheForcudnnConvolutionFwdAlgoPerf_t.Set(key, res);
  return res;
}

class AvgPool2DImplementedByCUDNNPoolingForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit AvgPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.avg_pool2d");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::PoolArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        args->include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, 2, BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.avg_pool2d"));
  }

 public:
  ~AvgPool2DImplementedByCUDNNPoolingForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::PoolArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DImplementedByCUDNNPoolingForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.avg_pool2d", AvgPool2DImplementedByCUDNNPoolingForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class AvgPool2DDxImplementedByCUDNNPoolingBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit AvgPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.avg_pool2d_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::PoolDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        args->include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, 2, BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.avg_pool2d_dx"));
  }

 public:
  ~AvgPool2DDxImplementedByCUDNNPoolingBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::PoolDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.avg_pool2d_dx", AvgPool2DDxImplementedByCUDNNPoolingBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  cudnnTensorDescriptor_t yDesc;
  double epsilon;
  double exponentialAverageFactor;
  explicit BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference(
      const CallValues& cv) {
    auto op = Op::Get("mnm.op.batch_norm_infer");
    this->arg_indices = {
        fschema_index[op]("x"),           fschema_index[op]("running_mean"),
        fschema_index[op]("running_var"), fschema_index[op]("w"),
        fschema_index[op]("b"),
    };
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, 1, 2, 3, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    epsilon = args->eps;
    exponentialAverageFactor = args->momentum;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.batch_norm_infer"));
  }

 public:
  ~BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* running_mean = args->running_mean;
    (void)running_mean;
    DLTensor* running_var = args->running_var;
    (void)running_var;
    DLTensor* b = args->b;
    (void)b;
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
MNM_OP_DISPATCH_PLEVEL("mnm.op.batch_norm_infer",
                       BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  cudnnTensorDescriptor_t yDesc;
  double epsilon;
  double exponentialAverageFactor;
  explicit BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining(const CallValues& cv) {
    auto op = Op::Get("mnm.op.batch_norm_train");
    this->arg_indices = {
        fschema_index[op]("x"),           fschema_index[op]("running_mean"),
        fschema_index[op]("running_var"), fschema_index[op]("w"),
        fschema_index[op]("b"),
    };
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
    auto yDesc_tt = SquashTensorShape(out0, {0, 1, 2, 3, out0->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    epsilon = args->eps;
    exponentialAverageFactor = args->momentum;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.batch_norm_train"));
  }

 public:
  ~BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* running_mean = args->running_mean;
    (void)running_mean;
    DLTensor* running_var = args->running_var;
    (void)running_var;
    DLTensor* b = args->b;
    (void)b;
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
MNM_OP_DISPATCH_PLEVEL("mnm.op.batch_norm_train",
                       BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnTensorDescriptor_t dBnScaleBiasDesc;
  double epsilon;
  explicit BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.batch_norm_train_dxwb");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::BatchNormTrainDxwbArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* out1 = tv->fields[1];
    (void)out1;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, 1, 2, 3, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out0, {0, 1, 2, 3, out0->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    auto dBnScaleBiasDesc_tt = SquashTensorShape(out1, {0, 0, 1, out1->ndim});
    dBnScaleBiasDesc = NormalizeTensorType(dBnScaleBiasDesc_tt);
    epsilon = args->eps;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.batch_norm_train_dxwb"));
  }

 public:
  ~BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dBnScaleBiasDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormTrainDxwbArgs>();
    (void)args;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    (void)x;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* out1 = tv->fields[1];
    (void)out1;
    DLTensor* w = args->w;
    (void)w;
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
MNM_OP_DISPATCH_PLEVEL("mnm.op.batch_norm_train_dxwb",
                       BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class Conv2DImplementedByCUDNNConvolutionForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DImplementedByCUDNNConvolutionForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.conv2d");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("w"),
    };
    auto args = cv->args.as<mnm::op::schema::ConvArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto wDesc_tt = SquashTensorShape(args->w, {});
    (void)wDesc_tt;
    wDesc = NormalizeFilter(args->w);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(w->dtype)));
    if (ir::DataType(w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    };
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << xDesc_tt
                << yDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionFwdAlgoPerf_tWrapper(algo_key, xDesc, wDesc, convDesc, yDesc);
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                       xDesc, wDesc, convDesc, yDesc, algo.algo,
                                                       &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.conv2d"));
  }

 public:
  ~Conv2DImplementedByCUDNNConvolutionForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::ConvArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* w = args->w;
    (void)w;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnConvolutionForward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), xDesc,
        x->data, wDesc, w->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 2);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* w = Downcast<TensorValue>(inputs[1]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnConvolutionForward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), xDesc,
        x->data, wDesc, w->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DImplementedByCUDNNConvolutionForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.conv2d", Conv2DImplementedByCUDNNConvolutionForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class Conv2DDwImplementedByCUDNNConvolutionBackwardFilter : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t dwDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdFilterAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DDwImplementedByCUDNNConvolutionBackwardFilter(const CallValues& cv) {
    auto op = Op::Get("mnm.op.conv2d_dw");
    this->arg_indices = {
        fschema_index[op]("x_or_w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::ConvDxwArgs>();
    (void)args;
    DLTensor* x_or_w = args->x_or_w;
    (void)x_or_w;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* dy = args->dy;
    (void)dy;
    auto xDesc_tt = SquashTensorShape(x_or_w, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto dwDesc_tt = SquashTensorShape(cv->out, {});
    (void)dwDesc_tt;
    dwDesc = NormalizeFilter(cv->out);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(x_or_w->dtype)));
    if (ir::DataType(x_or_w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    };
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << xDesc_tt << dyDesc_tt
                << dwDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo =
        FindcudnnConvolutionBwdFilterAlgoPerf_tWrapper(algo_key, xDesc, dyDesc, convDesc, dwDesc);
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, algo.algo,
        &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.conv2d_dw"));
  }

 public:
  ~Conv2DDwImplementedByCUDNNConvolutionBackwardFilter() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(dwDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::ConvDxwArgs>();
    (void)args;
    DLTensor* x_or_w = args->x_or_w;
    (void)x_or_w;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* dy = args->dy;
    (void)dy;

    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), xDesc,
        x_or_w->data, dyDesc, dy->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dwDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 2);
    DLTensor* x_or_w = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    DLTensor* dy = Downcast<TensorValue>(inputs[1]);
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), xDesc,
        x_or_w->data, dyDesc, dy->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dwDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DDwImplementedByCUDNNConvolutionBackwardFilter(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.conv2d_dw",
                       Conv2DDwImplementedByCUDNNConvolutionBackwardFilter::make, DevType::kCUDA(),
                       "generated_cudnn", 10);

class Conv2DDxImplementedByCUDNNConvolutionBackwardData : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t dxDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdDataAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DDxImplementedByCUDNNConvolutionBackwardData(const CallValues& cv) {
    auto op = Op::Get("mnm.op.conv2d_dx");
    this->arg_indices = {
        fschema_index[op]("x_or_w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::ConvDxwArgs>();
    (void)args;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* x_or_w = args->x_or_w;
    (void)x_or_w;
    DLTensor* dy = args->dy;
    (void)dy;
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    auto wDesc_tt = SquashTensorShape(args->x_or_w, {});
    (void)wDesc_tt;
    wDesc = NormalizeFilter(args->x_or_w);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(x_or_w->dtype)));
    if (ir::DataType(x_or_w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    };
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << dyDesc_tt
                << dxDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionBwdDataAlgoPerf_tWrapper(algo_key, wDesc, dyDesc, convDesc, dxDesc);
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                            wDesc, dyDesc, convDesc, dxDesc,
                                                            algo.algo, &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.conv2d_dx"));
  }

 public:
  ~Conv2DDxImplementedByCUDNNConvolutionBackwardData() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::ConvDxwArgs>();
    (void)args;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* x_or_w = args->x_or_w;
    (void)x_or_w;
    DLTensor* dy = args->dy;
    (void)dy;

    CUDNN_CALL(cudnnConvolutionBackwardData(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), wDesc,
        x_or_w->data, dyDesc, dy->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 2);
    DLTensor* out = Downcast<TensorValue>(output);
    DLTensor* x_or_w = Downcast<TensorValue>(inputs[0]);
    DLTensor* dy = Downcast<TensorValue>(inputs[1]);
    CUDNN_CALL(cudnnConvolutionBackwardData(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNNDType(out->dtype).const_addr<1>(), wDesc,
        x_or_w->data, dyDesc, dy->data, convDesc, algo.algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DDxImplementedByCUDNNConvolutionBackwardData(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.conv2d_dx", Conv2DDxImplementedByCUDNNConvolutionBackwardData::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class LogSoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;
  explicit LogSoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.log_softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.log_softmax"));
  }

 public:
  ~LogSoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG, mode,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG, mode,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.log_softmax", LogSoftmaxImplementedByCUDNNSoftmaxForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class LogSoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;
  explicit LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.log_softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, axis, axis + 1, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, axis, axis + 1, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.log_softmax_dx"));
  }

 public:
  ~LogSoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_LOG,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.log_softmax_dx", LogSoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 7);

class MaxPool2DImplementedByCUDNNPoolingForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit MaxPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.max_pool2d");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::PoolArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.max_pool2d"));
  }

 public:
  ~MaxPool2DImplementedByCUDNNPoolingForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::PoolArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingForward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                   CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DImplementedByCUDNNPoolingForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.max_pool2d", MaxPool2DImplementedByCUDNNPoolingForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class MaxPool2DDxImplementedByCUDNNPoolingBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit MaxPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.max_pool2d_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::PoolDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.max_pool2d_dx"));
  }

 public:
  ~MaxPool2DDxImplementedByCUDNNPoolingBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::PoolDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnPoolingBackward(CUDNNThreadEntry::ThreadLocal()->handle, poolingDesc,
                                    CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data, dyDesc,
                                    dy->data, xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.max_pool2d_dx", MaxPool2DDxImplementedByCUDNNPoolingBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class ReluImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit ReluImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.relu");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.relu"));
  }

 public:
  ~ReluImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new ReluImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.relu", ReluImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class ReluDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit ReluDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.relu_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.relu_dx"));
  }

 public:
  ~ReluDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new ReluDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.relu_dx", ReluDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class SigmoidImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit SigmoidImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.sigmoid");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.sigmoid"));
  }

 public:
  ~SigmoidImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new SigmoidImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.sigmoid", SigmoidImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class SigmoidDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit SigmoidDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.sigmoid_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.sigmoid_dx"));
  }

 public:
  ~SigmoidDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new SigmoidDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.sigmoid_dx", SigmoidDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class SoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;
  explicit SoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.softmax");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.softmax"));
  }

 public:
  ~SoftmaxImplementedByCUDNNSoftmaxForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                   mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                   mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                   CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.softmax", SoftmaxImplementedByCUDNNSoftmaxForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class SoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;
  explicit SoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.softmax_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    int axis = (args->axis + x->ndim) % x->ndim;
    auto xDesc_tt = SquashTensorShape(x, {0, axis, axis + 1, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, axis, axis + 1, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, axis, axis + 1, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, axis, axis + 1, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.softmax_dx"));
  }

 public:
  ~SoftmaxDxImplementedByCUDNNSoftmaxBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::SoftmaxDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnSoftmaxBackward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                    dyDesc, dy->data, CUDNNDType(out->dtype).const_addr<0>(),
                                    dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.softmax_dx", SoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 7);

class TanhImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit TanhImplementedByCUDNNActivationForward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.tanh");
    this->arg_indices = {
        fschema_index[op]("x"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.tanh"));
  }

 public:
  ~TanhImplementedByCUDNNActivationForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 1);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationForward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                      CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                      CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new TanhImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.tanh", TanhImplementedByCUDNNActivationForward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);

class TanhDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit TanhDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto op = Op::Get("mnm.op.tanh_dx");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("y"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;
    auto xDesc_tt = SquashTensorShape(x, {0, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(y, {0, y->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto dyDesc_tt = SquashTensorShape(dy, {0, dy->ndim});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    auto dxDesc_tt = SquashTensorShape(out, {0, out->ndim});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName("mnm.op.tanh_dx"));
  }

 public:
  ~TanhDxImplementedByCUDNNActivationBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activationDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    CHECK(args->x.defined());
    DLTensor* x = args->x.value();
    (void)x;
    CHECK(args->y.defined());
    DLTensor* y = args->y.value();
    (void)y;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out = cv->out;
    (void)out;

    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 3);
    DLTensor* x = Downcast<TensorValue>(inputs[0]);
    DLTensor* y = Downcast<TensorValue>(inputs[1]);
    DLTensor* dy = Downcast<TensorValue>(inputs[2]);
    DLTensor* out = Downcast<TensorValue>(output);
    CUDNN_CALL(cudnnActivationBackward(CUDNNThreadEntry::ThreadLocal()->handle, activationDesc,
                                       CUDNNDType(out->dtype).const_addr<1>(), yDesc, y->data,
                                       dyDesc, dy->data, xDesc, x->data,
                                       CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new TanhDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH_PLEVEL("mnm.op.tanh_dx", TanhDxImplementedByCUDNNActivationBackward::make,
                       DevType::kCUDA(), "generated_cudnn", 10);
}  // namespace generated
}  // namespace cudnn
}  // namespace op
}  // namespace mnm
