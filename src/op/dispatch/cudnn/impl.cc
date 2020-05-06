/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cudnn/impl.cc
 * \brief Operator schema. Auto generated. Do not touch.
 */
#include "../../op_utils.h"
#include "../../schema/gemm.h"
#include "../../schema/init.h"
#include "../../schema/likes.h"
#include "../../schema/list_args.h"
#include "../../schema/loss.h"
#include "../../schema/nn.h"
#include "../../schema/optimizer.h"
#include "../../schema/transform.h"
#include "../../schema/ufunc.h"
#include "./cudnn_utils.h"
namespace mnm {
namespace op {
namespace cudnn {
namespace generated {
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;
using common::shape_utils::PadDims;
using common::shape_utils::Shape2Strides;
using dmlc::BeginPtr;
using value::TupleValueObj;
MetaCache<cudnnConvolutionBwdDataAlgo_t> CacheForcudnnConvolutionBwdDataAlgo_t;
cudnnConvolutionBwdDataAlgo_t FindcudnnConvolutionBwdDataAlgo_tWrapper(
    const std::vector<uint8_t>& key, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc) {
  std::lock_guard<std::mutex> lock(CacheForcudnnConvolutionBwdDataAlgo_t.mu);
  if (auto* val = CacheForcudnnConvolutionBwdDataAlgo_t.Get(key)) {
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
  CacheForcudnnConvolutionBwdDataAlgo_t.Set(key, res.algo);
  return res.algo;
}
MetaCache<cudnnConvolutionBwdFilterAlgo_t> CacheForcudnnConvolutionBwdFilterAlgo_t;
cudnnConvolutionBwdFilterAlgo_t FindcudnnConvolutionBwdFilterAlgo_tWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc) {
  std::lock_guard<std::mutex> lock(CacheForcudnnConvolutionBwdFilterAlgo_t.mu);
  if (auto* val = CacheForcudnnConvolutionBwdFilterAlgo_t.Get(key)) {
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
  CacheForcudnnConvolutionBwdFilterAlgo_t.Set(key, res.algo);
  return res.algo;
}
MetaCache<cudnnConvolutionFwdAlgo_t> CacheForcudnnConvolutionFwdAlgo_t;
cudnnConvolutionFwdAlgo_t FindcudnnConvolutionFwdAlgo_tWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc) {
  std::lock_guard<std::mutex> lock(CacheForcudnnConvolutionFwdAlgo_t.mu);
  if (auto* val = CacheForcudnnConvolutionFwdAlgo_t.Get(key)) {
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
  CacheForcudnnConvolutionFwdAlgo_t.Set(key, res.algo);
  return res.algo;
}
class AvgPool2DImplementedByCUDNNPoolingForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit AvgPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
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
  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DImplementedByCUDNNPoolingForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.avg_pool2d", AvgPool2DImplementedByCUDNNPoolingForward::make,
                DevType::kCUDA(), "generated_cudnn");
class AvgPool2DDxImplementedByCUDNNPoolingBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit AvgPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
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
  static OpEnv* make(const CallValues& cv) {
    return new AvgPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.avg_pool2d_dx", AvgPool2DDxImplementedByCUDNNPoolingBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  explicit BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference(
      const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* w = args->w;
    (void)w;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out, {0, 1, 2, 3, out->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
  }

 public:
  ~BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out = cv->out;
    (void)out;
    DLTensor* w = args->w;
    (void)w;
    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out->data, bnScaleBiasMeanVarDesc, w->data, args->b->tensor->data,
        args->running_mean->tensor->data, args->running_var->tensor->data, args->eps));
  }
  static OpEnv* make(const CallValues& cv) {
    return new BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.batch_norm_infer",
                BatchNormInferImplementedByCUDNNBatchNormalizationForwardInference::make,
                DevType::kCUDA(), "generated_cudnn");
class BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  explicit BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    auto* tv = const_cast<TupleValueObj*>(cv->out.as<TupleValueObj>());
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* w = args->w;
    (void)w;
    auto xDesc_tt = SquashTensorShape(x, {0, 1, 2, 3, x->ndim});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto yDesc_tt = SquashTensorShape(out0, {0, 1, 2, 3, out0->ndim});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto bytes_of_out0 = BytesCompactTensor(*out0);
    RequestMemory(&out0->data, cv->ctx, bytes_of_out0);
    auto bnScaleBiasMeanVarDesc_tt = SquashTensorShape(w, {0, 0, 1, w->ndim});
    bnScaleBiasMeanVarDesc = NormalizeTensorType(bnScaleBiasMeanVarDesc_tt);
  }

 public:
  ~BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc));
  }
  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormArgs>();
    (void)args;
    auto* tv = const_cast<TupleValueObj*>(cv->out.as<TupleValueObj>());
    DLTensor* x = args->x;
    (void)x;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* w = args->w;
    (void)w;
    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(x->dtype).const_addr<1>(), CUDNNDType(x->dtype).const_addr<0>(), xDesc, x->data,
        yDesc, out0->data, bnScaleBiasMeanVarDesc, w->data, args->b->tensor->data, args->momentum,
        args->running_mean->tensor->data, args->running_var->tensor->data, args->eps, nullptr,
        nullptr));
  }
  static OpEnv* make(const CallValues& cv) {
    return new BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.batch_norm_train",
                BatchNormTrainImplementedByCUDNNBatchNormalizationForwardTraining::make,
                DevType::kCUDA(), "generated_cudnn");
class BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnTensorDescriptor_t dBnScaleBiasDesc;
  explicit BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::BatchNormTrainDxwbArgs>();
    (void)args;
    auto* tv = const_cast<TupleValueObj*>(cv->out.as<TupleValueObj>());
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
    auto bytes_of_out0 = BytesCompactTensor(*out0);
    RequestMemory(&out0->data, cv->ctx, bytes_of_out0);
    auto dBnScaleBiasDesc_tt = SquashTensorShape(out1, {0, 0, 1, out1->ndim});
    dBnScaleBiasDesc = NormalizeTensorType(dBnScaleBiasDesc_tt);
    auto bytes_of_out1 = BytesCompactTensor(*out1);
    RequestMemory(&out1->data, cv->ctx, bytes_of_out1);
    RequestMemory(&tv->fields[2].operator DLTensor*()->data, cv->ctx, bytes_of_out1);
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
    auto* tv = const_cast<TupleValueObj*>(cv->out.as<TupleValueObj>());
    DLTensor* x = args->x;
    (void)x;
    DLTensor* dy = args->dy;
    (void)dy;
    DLTensor* out0 = tv->fields[0];
    (void)out0;
    DLTensor* out1 = tv->fields[1];
    (void)out1;
    CUDNN_CALL(cudnnBatchNormalizationBackward(
        CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_BATCHNORM_SPATIAL,
        CUDNNDType(out0->dtype).const_addr<1>(), CUDNNDType(out0->dtype).const_addr<0>(),
        CUDNNDType(out1->dtype).const_addr<1>(), CUDNNDType(out1->dtype).const_addr<0>(), xDesc,
        x->data, dyDesc, dy->data, dxDesc, out0->data, dBnScaleBiasDesc, args->w->tensor->data,
        out1->data, tv->fields[2].operator DLTensor*()->data, args->eps, nullptr, nullptr));
  }
  static OpEnv* make(const CallValues& cv) {
    return new BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.batch_norm_train_dxwb",
                BatchNormTrainDxwbImplementedByCUDNNBatchNormalizationBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class Conv2DImplementedByCUDNNConvolutionForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DImplementedByCUDNNConvolutionForward(const CallValues& cv) {
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
    auto wDesc_tt = SquashTensorShape(w, {});
    (void)wDesc_tt;
    wDesc = NormalizeFilter(w);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(w->dtype)));
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << xDesc_tt
                << yDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionFwdAlgo_tWrapper(algo_key, xDesc, wDesc, convDesc, yDesc);
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                       xDesc, wDesc, convDesc, yDesc, algo,
                                                       &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->ctx, workSpaceSizeInBytes);
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
        x->data, wDesc, w->data, convDesc, algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DImplementedByCUDNNConvolutionForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.conv2d", Conv2DImplementedByCUDNNConvolutionForward::make, DevType::kCUDA(),
                "generated_cudnn");
class Conv2DDwImplementedByCUDNNConvolutionBackwardFilter : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t dwDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdFilterAlgo_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DDwImplementedByCUDNNConvolutionBackwardFilter(const CallValues& cv) {
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
    auto dwDesc_tt = SquashTensorShape(out, {});
    (void)dwDesc_tt;
    dwDesc = NormalizeFilter(out);
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(x_or_w->dtype)));
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << xDesc_tt << dyDesc_tt
                << dwDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionBwdFilterAlgo_tWrapper(algo_key, xDesc, dyDesc, convDesc, dwDesc);
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, algo,
        &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->ctx, workSpaceSizeInBytes);
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
        x_or_w->data, dyDesc, dy->data, convDesc, algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dwDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DDwImplementedByCUDNNConvolutionBackwardFilter(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.conv2d_dw", Conv2DDwImplementedByCUDNNConvolutionBackwardFilter::make,
                DevType::kCUDA(), "generated_cudnn");
class Conv2DDxImplementedByCUDNNConvolutionBackwardData : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t dxDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdDataAlgo_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;
  explicit Conv2DDxImplementedByCUDNNConvolutionBackwardData(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    auto wDesc_tt = SquashTensorShape(x_or_w, {});
    (void)wDesc_tt;
    wDesc = NormalizeFilter(x_or_w);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               CUDNNDType(x_or_w->dtype)));
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << dyDesc_tt
                << dxDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionBwdDataAlgo_tWrapper(algo_key, wDesc, dyDesc, convDesc, dxDesc);
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                            wDesc, dyDesc, convDesc, dxDesc, algo,
                                                            &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->ctx, workSpaceSizeInBytes);
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
        x_or_w->data, dyDesc, dy->data, convDesc, algo, workSpace, workSpaceSizeInBytes,
        CUDNNDType(out->dtype).const_addr<0>(), dxDesc, out->data));
  }
  static OpEnv* make(const CallValues& cv) {
    return new Conv2DDxImplementedByCUDNNConvolutionBackwardData(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.conv2d_dx", Conv2DDxImplementedByCUDNNConvolutionBackwardData::make,
                DevType::kCUDA(), "generated_cudnn");
class LogSoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;
  explicit LogSoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
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
  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.log_softmax", LogSoftmaxImplementedByCUDNNSoftmaxForward::make,
                DevType::kCUDA(), "generated_cudnn");
class LogSoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;
  explicit LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
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
  static OpEnv* make(const CallValues& cv) {
    return new LogSoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.log_softmax_dx", LogSoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class MaxPool2DImplementedByCUDNNPoolingForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit MaxPool2DImplementedByCUDNNPoolingForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
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
  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DImplementedByCUDNNPoolingForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.max_pool2d", MaxPool2DImplementedByCUDNNPoolingForward::make,
                DevType::kCUDA(), "generated_cudnn");
class MaxPool2DDxImplementedByCUDNNPoolingBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnPoolingDescriptor_t poolingDesc;
  explicit MaxPool2DDxImplementedByCUDNNPoolingBackward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    std::vector<int> kernel = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->kernel));
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CALL(cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2,
                                           BeginPtr(kernel), BeginPtr(padding), BeginPtr(stride)));
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
  static OpEnv* make(const CallValues& cv) {
    return new MaxPool2DDxImplementedByCUDNNPoolingBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.max_pool2d_dx", MaxPool2DDxImplementedByCUDNNPoolingBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class ReluImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit ReluImplementedByCUDNNActivationForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
  static OpEnv* make(const CallValues& cv) {
    return new ReluImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.relu", ReluImplementedByCUDNNActivationForward::make, DevType::kCUDA(),
                "generated_cudnn");
class ReluDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit ReluDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
  static OpEnv* make(const CallValues& cv) {
    return new ReluDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.relu_dx", ReluDxImplementedByCUDNNActivationBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class SigmoidImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit SigmoidImplementedByCUDNNActivationForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
  static OpEnv* make(const CallValues& cv) {
    return new SigmoidImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.sigmoid", SigmoidImplementedByCUDNNActivationForward::make,
                DevType::kCUDA(), "generated_cudnn");
class SigmoidDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit SigmoidDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
  static OpEnv* make(const CallValues& cv) {
    return new SigmoidDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.sigmoid_dx", SigmoidDxImplementedByCUDNNActivationBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class SoftmaxImplementedByCUDNNSoftmaxForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnSoftmaxMode_t mode;
  explicit SoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
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
  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxImplementedByCUDNNSoftmaxForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.softmax", SoftmaxImplementedByCUDNNSoftmaxForward::make, DevType::kCUDA(),
                "generated_cudnn");
class SoftmaxDxImplementedByCUDNNSoftmaxBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnSoftmaxMode_t mode;
  explicit SoftmaxDxImplementedByCUDNNSoftmaxBackward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    mode = GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1
               ? CUDNN_SOFTMAX_MODE_INSTANCE
               : CUDNN_SOFTMAX_MODE_CHANNEL;
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
  static OpEnv* make(const CallValues& cv) {
    return new SoftmaxDxImplementedByCUDNNSoftmaxBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.softmax_dx", SoftmaxDxImplementedByCUDNNSoftmaxBackward::make,
                DevType::kCUDA(), "generated_cudnn");
class TanhImplementedByCUDNNActivationForward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit TanhImplementedByCUDNNActivationForward(const CallValues& cv) {
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
  static OpEnv* make(const CallValues& cv) {
    return new TanhImplementedByCUDNNActivationForward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.tanh", TanhImplementedByCUDNNActivationForward::make, DevType::kCUDA(),
                "generated_cudnn");
class TanhDxImplementedByCUDNNActivationBackward : public mnm::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnTensorDescriptor_t dxDesc;
  cudnnActivationDescriptor_t activationDesc;
  explicit TanhDxImplementedByCUDNNActivationBackward(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::UnaryDxArgs>();
    (void)args;
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
    auto bytes_of_out = BytesCompactTensor(*out);
    RequestMemory(&out->data, cv->ctx, bytes_of_out);
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH,
                                            CUDNN_PROPAGATE_NAN, 0.0));
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
    DLTensor* x = args->x;
    (void)x;
    DLTensor* y = args->y;
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
  static OpEnv* make(const CallValues& cv) {
    return new TanhDxImplementedByCUDNNActivationBackward(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.tanh_dx", TanhDxImplementedByCUDNNActivationBackward::make,
                DevType::kCUDA(), "generated_cudnn");
}  // namespace generated
}  // namespace cudnn
}  // namespace op
}  // namespace mnm
