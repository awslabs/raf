#include <mnm/op.h>

#include "../../../common/arg_utils.h"
#include "../../../common/cuda.h"
#include "../../../common/shape_utils.h"
#include "../../attrs/conv.h"
#include "./util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {
namespace manual {

using common::shape_utils::MakeShape;
using rly::Array;
using rly::Attrs;
using value::Value;

class ConvolutionModeEnum final
    : public EnumBase<ConvolutionModeEnum, 2, int32_t, cudnnConvolutionMode_t> {
 public:
  ENUM_DEF_HEADER(ConvolutionModeEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionModeEnum, 0, Convolution, CUDNN_CONVOLUTION,
                           "CUDNN_CONVOLUTION");
  ENUM_DEF_ENTRY_WITH_NAME(ConvolutionModeEnum, 1, CrossCorrelation, CUDNN_CROSS_CORRELATION,
                           "CUDNN_CROSS_CORRELATION");
};

class Conv2dCUDNN : public mnm::op::OpEnv {
 public:
  DType dtype;
  Context ctx;
  cudnnHandle_t handle;
  const void* alpha;
  cudnnTensorDescriptor_t in_desc;
  cudnnFilterDescriptor_t ker_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t algo;
  size_t ws_size;
  void* ws;
  const void* beta;
  cudnnTensorDescriptor_t out_desc;
  void* out_data;

  Conv2dCUDNN() {
    handle = CUDNNThreadEntry::ThreadLocal()->handle;
  }
  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs);
  void RequestMemory(void** dest, Context ctx, int64_t nb);
  void RequestWorkspace(void** dest, Context ctx, int64_t nb);

  void Execute(rly::Array<value::Value> args, rly::Attrs) override final;

  ~Conv2dCUDNN() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(ker_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<Conv2dCUDNN> res = std::make_unique<Conv2dCUDNN>();
    res->PreAllocate(args, attrs);
    return res.release();
  }
};

void Conv2dCUDNN::Execute(Array<Value> args, Attrs) {
  const DLTensor* in = args[0];
  const DLTensor* fil = args[0];
  CUDNN_CALL(cudnnConvolutionForward(handle, alpha, in_desc, in->data, ker_desc, fil->data,
                                     conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, ws, ws_size,
                                     beta, out_desc, out_data));
}

AlgorithmCache<cudnnConvolutionFwdAlgo_t> _conv_fwd_alg_cache;

cudnnConvolutionFwdAlgo_t FindConvolutionForwardAlgorithm(const std::vector<int>& key,
                                                          cudnnTensorDescriptor_t xDesc,
                                                          cudnnFilterDescriptor_t wDesc,
                                                          cudnnConvolutionDescriptor_t convDesc,
                                                          cudnnTensorDescriptor_t yDesc) {
  if (_conv_fwd_alg_cache.has(key)) {
    return _conv_fwd_alg_cache.get(key);
  }
  int cnt;
  cudnnConvolutionFwdAlgoPerf_t res;
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(CUDNNThreadEntry::ThreadLocal()->handle, xDesc,
                                                  wDesc, convDesc, yDesc, 1, &cnt, &res));
  _conv_fwd_alg_cache.set(key, res.algo);
  return res.algo;
}

void Conv2dCUDNN::PreAllocate(Array<Value> args, Attrs attrs) {
  CUDNN_CALL(cudnnCreate(&handle));
  const auto dlts = common::arg_utils::AsVector(args);
  dtype = common::arg_utils::DeduceDLType(dlts);
  ctx = common::arg_utils::DeduceCtx(dlts);

  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));

  FORM_SHAPE(in_shape, dlts[0]);
  FORM_STRIDE(in_stride, in_shape);
  const DLTensor* dlt0 = args[0];
  CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc, CUDNNDType(dtype), in_shape.size(),
                                        dmlc::BeginPtr(in_shape), dmlc::BeginPtr(in_stride)));

  CUDNN_CALL(cudnnCreateFilterDescriptor(&ker_desc));
  FORM_SHAPE(ker_shape, dlts[1]);
  CUDNN_CALL(cudnnSetFilterNdDescriptor(ker_desc, CUDNNDType(dtype), CUDNN_TENSOR_NCHW,
                                        ker_shape.size(), dmlc::BeginPtr(ker_shape)));

  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  auto c2a = attrs.as<mnm::op::attrs::Conv2DAttrs>();
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(
      conv_desc, c2a->padding.size(), dmlc::BeginPtr(MakeShape<int>(c2a->padding)),
      dmlc::BeginPtr(MakeShape<int>(c2a->stride)), dmlc::BeginPtr(MakeShape<int>(c2a->dilation)),
      CUDNN_CONVOLUTION, CUDNNDType(dtype)));

  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  int ndims;
  ndims = ker_shape.size();
  std::vector<int> out_shape(ndims);
  CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, ker_desc, ndims,
                                                   dmlc::BeginPtr(out_shape)));

  FORM_STRIDE(out_stride, out_shape);
  int out_size;
  out_size = out_stride[0] * out_shape[0];
  RequestMemory(&out_data, Context(DevType::kCUDA(), 0), (out_size * dlt0->dtype.bits + 7) / 8);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc, CUDNNDType(dtype), ndims,
                                        dmlc::BeginPtr(out_shape), dmlc::BeginPtr(out_stride)));

  auto concat_vecs = ConcatVecs(in_shape, in_stride, ker_shape, c2a->padding, c2a->stride,
                                c2a->dilation, out_shape, out_stride);

  auto algo = FindConvolutionForwardAlgorithm(concat_vecs, in_desc, ker_desc, conv_desc, out_desc);
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, ker_desc, conv_desc, out_desc,
                                                     CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                                                     &ws_size));
  RequestWorkspace(&ws, Context(DevType::kCUDA(), 0), ws_size);

  alpha = CUDNNDType(dtype).const_addr<1>();
  beta = CUDNNDType(dtype).const_addr<0>();
}

void Conv2dCUDNN::RequestMemory(void** dest, Context ctx, int64_t nb) {
  CUDA_CALL(cudaMalloc(dest, nb));
}

void Conv2dCUDNN::RequestWorkspace(void** dest, Context ctx, int64_t nb) {
  CUDA_CALL(cudaMalloc(dest, nb));
}

MNM_REGISTER_OP_DISPATCH("mnm.op.conv2d", DevType::kCUDA(), "manual_cudnn", Conv2dCUDNN::make);

}  // namespace manual
}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
