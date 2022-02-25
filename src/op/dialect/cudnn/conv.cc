/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/conv.cc
 * \brief CUDNN conv2d operators.
 */
#include <queue>
#include "../../schema/nn.h"
#include "../3rdparty/tvm/src/runtime/file_utils.h"
#include "./cudnn_utils.h"
#include "raf/ir.h"
#include "raf/memory_pool.h"
#include "raf/op_utils.h"

namespace raf {
namespace op {
namespace cudnn {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::memory_pool;
using dmlc::BeginPtr;

static auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");

template <typename T>
class CuDNNConvAlgoCacheEntry {
 public:
  CuDNNConvAlgoCacheEntry(T algo_perf) : algo_perf_(algo_perf) {
  }

  T Value() const {
    return algo_perf_;
  }

  static CuDNNConvAlgoCacheEntry Load(const std::string& path) {
    std::string data;
    tvm::runtime::LoadBinaryFromFile(path + "/value.bin", &data);
    dmlc::MemoryStringStream reader(&data);
    dmlc::Stream* stream = &reader;

    T algo_perf;
    stream->Read(&algo_perf.algo);
    stream->Read(&algo_perf.status);
    stream->Read(&algo_perf.time);
    stream->Read(&algo_perf.memory);
    stream->Read(&algo_perf.determinism);
    stream->Read(&algo_perf.mathType);
    stream->Read(algo_perf.reserved);
    return CuDNNConvAlgoCacheEntry(algo_perf);
  }

  bool Save(const std::string& path) {
    std::string data;
    dmlc::MemoryStringStream writer(&data);
    dmlc::SeekStream* stream = &writer;
    stream->Write(algo_perf_.algo);
    stream->Write(algo_perf_.status);
    stream->Write(algo_perf_.time);
    stream->Write(algo_perf_.memory);
    stream->Write(algo_perf_.determinism);
    stream->Write(algo_perf_.mathType);
    stream->Write(algo_perf_.reserved);
    tvm::runtime::SaveBinaryToFile(path + "/value.bin", data);
    return true;
  }

 private:
  T algo_perf_;
};

template <class Algo, class F>
void GetMaxWorkspaceSize(const Algo* algos, int n_algos, F fget_workspace, size_t* max_ws_size,
                         std::shared_ptr<Memory>* memory, const Device& device) {
  std::priority_queue<size_t> max_ws_sizes;
  for (int i = 0; i < n_algos; ++i) {
    size_t ws_size = 0;
    try {
      fget_workspace(algos[i], &ws_size);
    } catch (const dmlc::Error& e) {
      continue;
    }
    max_ws_sizes.push(ws_size);
  }
  while (!max_ws_sizes.empty()) {
    try {
      size_t size = max_ws_sizes.top();
      max_ws_sizes.pop();
      *memory = Memory::Alloc(device, size);
      *max_ws_size = size;
      break;
    } catch (const dmlc::Error& e) {
      continue;
    }
  }
  CHECK(*memory != nullptr);
}

MetaPersistCache<CuDNNConvAlgoCacheEntry<cudnnConvolutionFwdAlgoPerf_t>> CacheCudnnConvFwdAlgoPerf(
    "cudnn_conv_fwd_algo_perf");

cudnnConvolutionFwdAlgoPerf_t FindcudnnConvolutionFwdAlgoPerf_tExWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void* y, const Device& device) {
  if (auto* val = CacheCudnnConvFwdAlgoPerf.Get(key)) {
    return val->Value();
  }
  static const cudnnConvolutionFwdAlgo_t algos[] = {
      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};
  constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  cudnnConvolutionFwdAlgoPerf_t res[num_algos];
  int cnt;
  if (CUDNNThreadEntry::ThreadLocal()->benchmark) {
    size_t max_ws_size;
    std::shared_ptr<Memory> memory;
    auto fget_workspace = [&](cudnnConvolutionFwdAlgo_t algo, size_t* ws_size) {
      CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          CUDNNThreadEntry::ThreadLocal()->handle, xDesc, wDesc, convDesc, yDesc, algo, ws_size));
    };
    GetMaxWorkspaceSize(algos, sizeof(algos) / sizeof(cudnnConvolutionFwdAlgo_t), fget_workspace,
                        &max_ws_size, &memory, device);
    void* workspace_temp = memory->data;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, x, wDesc, w, convDesc, yDesc, y, num_algos,
        &cnt, res, workspace_temp, max_ws_size));
  } else {
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(CUDNNThreadEntry::ThreadLocal()->handle,
                                                      xDesc, wDesc, convDesc, yDesc, num_algos,
                                                      &cnt, res));
  }
  if (res[0].status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm "
               << cudnnGetErrorString(res[0].status);
    throw;
  }
  CacheCudnnConvFwdAlgoPerf.Set(key,
                                CuDNNConvAlgoCacheEntry<cudnnConvolutionFwdAlgoPerf_t>(res[0]));
  // debug information
  auto best_algo = res[0].algo;
  DLOG(INFO) << "CUDNN Found " << cnt << " conv2d algorithms, choosing "
             << conv2dFwdAlgoToString(best_algo);
  for (int i = 0; i < cnt; ++i) {
    DLOG(INFO) << "    " << i << ") " << conv2dFwdAlgoToString(res[i].algo)
               << " - time: " << res[i].time << " ms"
               << ", memory: " << res[i].memory
               << ", math type: " << cudnnMathTypeToString(res[i].mathType)
               << ", status: " << cudnnGetErrorString(res[i].status);
  }
  return res[0];
}

MetaPersistCache<CuDNNConvAlgoCacheEntry<cudnnConvolutionBwdDataAlgoPerf_t>>
    CacheCudnnConvBwdDataAlgoPerf("cudnn_conv_bwd_data_algo_perf");

cudnnConvolutionBwdDataAlgoPerf_t FindcudnnConvolutionBwdDataAlgoPerf_tExWrapper(
    const std::vector<uint8_t>& key, const cudnnFilterDescriptor_t wDesc, const void* w,
    const cudnnTensorDescriptor_t dyDesc, const void* dy,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void* dx,
    const Device& device) {
  if (auto* val = CacheCudnnConvBwdDataAlgoPerf.Get(key)) {
    return val->Value();
  }
  static const cudnnConvolutionBwdDataAlgo_t algos[] = {
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
  constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
  cudnnConvolutionBwdDataAlgoPerf_t res[num_algos];
  int cnt;
  if (CUDNNThreadEntry::ThreadLocal()->benchmark) {
    size_t max_ws_size;
    std::shared_ptr<Memory> memory;
    auto fget_workspace = [&](cudnnConvolutionBwdDataAlgo_t algo, size_t* ws_size) {
      CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          CUDNNThreadEntry::ThreadLocal()->handle, wDesc, dyDesc, convDesc, dxDesc, algo, ws_size));
    };
    GetMaxWorkspaceSize(algos, sizeof(algos) / sizeof(cudnnConvolutionBwdDataAlgo_t),
                        fget_workspace, &max_ws_size, &memory, device);
    void* workspace_temp = memory->data;
    CUDNN_CALL(cudnnFindConvolutionBackwardDataAlgorithmEx(
        CUDNNThreadEntry::ThreadLocal()->handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
        num_algos, &cnt, res, workspace_temp, max_ws_size));
  } else {
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(CUDNNThreadEntry::ThreadLocal()->handle,
                                                           wDesc, dyDesc, convDesc, dxDesc,
                                                           num_algos, &cnt, res));
  }
  if (res[0].status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm "
               << cudnnGetErrorString(res[0].status);
    throw;
  }
  auto best_algo = res[0].algo;
  CacheCudnnConvBwdDataAlgoPerf.Set(
      key, CuDNNConvAlgoCacheEntry<cudnnConvolutionBwdDataAlgoPerf_t>(res[0]));
  // debug information
  DLOG(INFO) << "CUDNN Found " << cnt << " conv2d_dx algorithms , choosing "
             << conv2dBwdDataAlgoToString(best_algo);
  for (int i = 0; i < cnt; ++i) {
    DLOG(INFO) << "    " << i << ") " << conv2dBwdDataAlgoToString(res[i].algo)
               << " - time: " << res[i].time << " ms"
               << ", memory: " << res[i].memory
               << ", math type: " << cudnnMathTypeToString(res[i].mathType)
               << ", status: " << cudnnGetErrorString(res[i].status);
  }
  return res[0];
}

MetaPersistCache<CuDNNConvAlgoCacheEntry<cudnnConvolutionBwdFilterAlgoPerf_t>>
    CacheCudnnConvBwdFilterAlgoPerf("cudnn_conv_bwd_filter_algo_perf");

cudnnConvolutionBwdFilterAlgoPerf_t FindcudnnConvolutionBwdFilterAlgoPerf_tExWrapper(
    const std::vector<uint8_t>& key, const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnTensorDescriptor_t dyDesc, const void* dy,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void* dw,
    const Device& device) {
  if (auto* val = CacheCudnnConvBwdFilterAlgoPerf.Get(key)) {
    return val->Value();
  }
  static const cudnnConvolutionBwdFilterAlgo_t algos[] = {
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED};
  constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
  cudnnConvolutionBwdFilterAlgoPerf_t res[num_algos];
  int cnt;
  if (CUDNNThreadEntry::ThreadLocal()->benchmark) {
    size_t max_ws_size;
    std::shared_ptr<Memory> memory;
    auto fget_workspace = [&](cudnnConvolutionBwdFilterAlgo_t algo, size_t* ws_size) {
      CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, algo, ws_size));
    };
    GetMaxWorkspaceSize(algos, sizeof(algos) / sizeof(cudnnConvolutionBwdFilterAlgo_t),
                        fget_workspace, &max_ws_size, &memory, device);
    void* workspace_temp = memory->data;
    CUDNN_CALL(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw,
        num_algos, &cnt, res, workspace_temp, max_ws_size));
  } else {
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, num_algos, &cnt,
        res));
  }
  if (res[0].status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm "
               << cudnnGetErrorString(res[0].status);
    throw;
  }
  CacheCudnnConvBwdFilterAlgoPerf.Set(
      key, CuDNNConvAlgoCacheEntry<cudnnConvolutionBwdFilterAlgoPerf_t>(res[0]));
  // debug information
  auto best_algo = res[0].algo;
  DLOG(INFO) << "CUDNN Found " << cnt << " conv2d_dw algorithms , choosing "
             << conv2dBwdFilterAlgoToString(best_algo);
  for (int i = 0; i < cnt; ++i) {
    DLOG(INFO) << "    " << i << ") " << conv2dBwdFilterAlgoToString(res[i].algo)
               << " - time: " << res[i].time << " ms"
               << ", memory: " << res[i].memory
               << ", math type: " << cudnnMathTypeToString(res[i].mathType)
               << ", status: " << cudnnGetErrorString(res[i].status);
  }
  return res[0];
}

class Conv2DImplementedByCUDNNConvolutionForward : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t yDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;

  explicit Conv2DImplementedByCUDNNConvolutionForward(const CallValues& cv) {
    auto op = Op::Get("raf.op.conv2d");
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("w"),
    };
    auto args = cv->args.as<raf::op::schema::ConvArgs>();
    DLTensor* x = args->x;
    DLTensor* w = args->w;
    DLTensor* out = cv->out;
    auto xDesc_tt = SquashTensorShape(x, {});
    xDesc = NormalizeTensorType(xDesc_tt);
    auto wDesc_tt = SquashTensorShape(args->w, {});
    wDesc = NormalizeFilter(args->w);
    auto yDesc_tt = SquashTensorShape(out, {});
    yDesc = NormalizeTensorType(yDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    cudnnDataType_t conv_dtype = CUDNNDType(w->dtype);
    // Use data type fp32 in the convolution descriptor when data type is fp16
    if (conv_dtype == CUDNN_DATA_HALF) conv_dtype = CUDNN_DATA_FLOAT;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               conv_dtype));
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    if (ir::DataType(w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    }

    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << xDesc_tt
                << yDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionFwdAlgoPerf_tExWrapper(algo_key, xDesc, x->data, wDesc, w->data,
                                                      convDesc, yDesc, out->data, cv->device);
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                       xDesc, wDesc, convDesc, yDesc, algo.algo,
                                                       &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
  }

 public:
  ~Conv2DImplementedByCUDNNConvolutionForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.conv2d"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::ConvArgs>();
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

RAF_REGISTER_DIALECT_OP(cudnn, conv2d, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.conv2d", Conv2DImplementedByCUDNNConvolutionForward::make);

class Conv2DDwImplementedByCUDNNConvolutionBackwardFilter : public raf::op::OpEnv {
  cudnnTensorDescriptor_t xDesc;
  cudnnFilterDescriptor_t dwDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdFilterAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;

  explicit Conv2DDwImplementedByCUDNNConvolutionBackwardFilter(const CallValues& cv) {
    auto op = Op::Get("raf.op.conv2d_dw");
    this->arg_indices = {
        fschema_index[op]("x_or_w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::ConvDxwArgs>();
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
    cudnnDataType_t conv_dtype = CUDNNDType(x_or_w->dtype);
    // Use data type fp32 in the convolution descriptor when data type is fp16
    if (conv_dtype == CUDNN_DATA_HALF) conv_dtype = CUDNN_DATA_FLOAT;
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               conv_dtype));
    if (ir::DataType(x_or_w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    };
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << xDesc_tt << dyDesc_tt
                << dwDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionBwdFilterAlgoPerf_tExWrapper(
        algo_key, xDesc, x_or_w->data, dyDesc, dy->data, convDesc, dwDesc, out->data, cv->device);
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        CUDNNThreadEntry::ThreadLocal()->handle, xDesc, dyDesc, convDesc, dwDesc, algo.algo,
        &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
  }

 public:
  ~Conv2DDwImplementedByCUDNNConvolutionBackwardFilter() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(dwDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.conv2d_dw"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::ConvDxwArgs>();
    DLTensor* x_or_w = args->x_or_w;
    DLTensor* out = cv->out;
    DLTensor* dy = args->dy;
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

RAF_REGISTER_DIALECT_OP(cudnn, conv2d_dw, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.conv2d_dw",
                 Conv2DDwImplementedByCUDNNConvolutionBackwardFilter::make);

class Conv2DDxImplementedByCUDNNConvolutionBackwardData : public raf::op::OpEnv {
  cudnnTensorDescriptor_t dxDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnTensorDescriptor_t dyDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionBwdDataAlgoPerf_t algo;
  size_t workSpaceSizeInBytes;
  void* workSpace;

  explicit Conv2DDxImplementedByCUDNNConvolutionBackwardData(const CallValues& cv) {
    auto op = Op::Get("raf.op.conv2d_dx");
    this->arg_indices = {
        fschema_index[op]("x_or_w"),
        fschema_index[op]("dy"),
    };
    auto args = cv->args.as<raf::op::schema::ConvDxwArgs>();
    DLTensor* out = cv->out;
    DLTensor* x_or_w = args->x_or_w;
    DLTensor* dy = args->dy;
    auto dxDesc_tt = SquashTensorShape(out, {});
    dxDesc = NormalizeTensorType(dxDesc_tt);
    auto wDesc_tt = SquashTensorShape(args->x_or_w, {});
    wDesc = NormalizeFilter(args->x_or_w);
    auto dyDesc_tt = SquashTensorShape(dy, {});
    dyDesc = NormalizeTensorType(dyDesc_tt);
    std::vector<int> stride = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->stride));
    std::vector<int> padding = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->padding));
    std::vector<int> dilation = CastVector<int, int64_t>(NormalizeScalarToTuple<2>(args->dilation));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    cudnnDataType_t conv_dtype = CUDNNDType(x_or_w->dtype);
    // Use data type fp32 in the convolution descriptor when data type is fp16
    if (conv_dtype == CUDNN_DATA_HALF) conv_dtype = CUDNN_DATA_FLOAT;
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(convDesc, 2, BeginPtr(padding), BeginPtr(stride),
                                               BeginPtr(dilation), CUDNN_CROSS_CORRELATION,
                                               conv_dtype));
    if (ir::DataType(x_or_w->dtype).is_float16()) {
      cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    };
    cudnnSetConvolutionGroupCount(convDesc, args->groups);
    HashKey algo_hasher;
    algo_hasher << args->stride << args->padding << args->dilation << wDesc_tt << dyDesc_tt
                << dxDesc_tt;
    const auto& algo_key = algo_hasher.byte_vector;
    algo = FindcudnnConvolutionBwdDataAlgoPerf_tExWrapper(
        algo_key, wDesc, x_or_w->data, dyDesc, dy->data, convDesc, dxDesc, out->data, cv->device);
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                            wDesc, dyDesc, convDesc, dxDesc,
                                                            algo.algo, &workSpaceSizeInBytes));
    RequestWorkspace(&workSpace, cv->device, workSpaceSizeInBytes);
    cudnnSetConvolutionMathType(convDesc, algo.mathType);
  }

 public:
  ~Conv2DDxImplementedByCUDNNConvolutionBackwardData() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn.conv2d_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::ConvDxwArgs>();
    DLTensor* out = cv->out;
    DLTensor* x_or_w = args->x_or_w;
    DLTensor* dy = args->dy;
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

RAF_REGISTER_DIALECT_OP(cudnn, conv2d_dx, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn.conv2d_dx", Conv2DDxImplementedByCUDNNConvolutionBackwardData::make);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
