#include <mnm/op.h>

#include "../../attrs/conv.h"
#include "./util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {
namespace manual {

using op::backend::cudnn::getAlphaBeta;
using rly::Array;
using rly::Attrs;
using value::Value;

class Conv2dCUDNN : public mnm::op::OpEnv {
 public:
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
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyFilterDescriptor(ker_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
  }

  static OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<Conv2dCUDNN> res(new Conv2dCUDNN());
    res->PreAllocate(args, attrs);
    return res.release();
  }
};

void Conv2dCUDNN::Execute(Array<Value> args, Attrs) {
  const DLTensor* in = args[0];
  const DLTensor* fil = args[0];
  CUDNN_CALL(cudnnConvolutionForward(handle, alpha, in_desc, in->data, ker_desc, fil->data,
                                     conv_desc, algo, ws, ws_size, beta, out_desc, out_data));
}

void Conv2dCUDNN::PreAllocate(Array<Value> args, Attrs attrs) {
  cudnnDataType_t dt;

  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  static cudnnMoreThan4DTensor in_tensor;
  const DLTensor* dlt0 = args[0];
  in_tensor = cudnnMoreThan4DTensor(dlt0);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc, in_tensor.dt, in_tensor.n, in_tensor.dims,
                                        in_tensor.strides));

  CUDNN_CALL(cudnnCreateFilterDescriptor(&ker_desc));
  static cudnnMoreThan4DTensor ker_tensor;
  const DLTensor* dlt1 = args[1];
  ker_tensor = cudnnMoreThan4DTensor(dlt1);
  CUDNN_CALL(cudnnSetFilterNdDescriptor(ker_desc, ker_tensor.dt, CUDNN_TENSOR_NCHW, ker_tensor.n,
                                        ker_tensor.dims));

  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  auto c2a = attrs.as<mnm::op::attrs::Conv2DAttrs>();
  int padding[8];
  ArrayIntegerToIntPtr(c2a->padding, padding);
  int striding[8];
  ArrayIntegerToIntPtr(c2a->stride, striding);
  int dilation[8];
  ArrayIntegerToIntPtr(c2a->dilation, dilation);

  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(conv_desc, c2a->padding.size(), padding, striding,
                                             dilation, CUDNN_CONVOLUTION, dt));

  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  static int n, out_size;
  n = ker_tensor.n;
  static int dims[8];
  static int strides[8];
  CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(conv_desc, in_desc, ker_desc, n, dims));
  out_size = MakeStride(n, dims, strides);

  RequestMemory(&out_data, Context(DevType::kCUDA(), 0), (out_size * dlt0->dtype.bits + 7) / 8);
  CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc, dt, n, dims, strides));

  algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, ker_desc, conv_desc, out_desc,
                                                     algo, &ws_size));
  RequestWorkspace(&ws, Context(DevType::kCUDA(), 0), ws_size);

  alpha = getAlphaBeta<1>(in_tensor.dt);
  beta = getAlphaBeta<0>(in_tensor.dt);
}

void Conv2dCUDNN::RequestMemory(void** dest, Context ctx, int64_t nb) {
  assert(cudaMalloc(dest, nb) == cudaSuccess);
}

void Conv2dCUDNN::RequestWorkspace(void** dest, Context ctx, int64_t nb) {
  assert(cudaMalloc(dest, nb) == cudaSuccess);
}

MNM_REGISTER_OP_DISPATCH("mnm.conv2d", DevType::kCUDA(), "test_cudnn", Conv2dCUDNN::make);

}  // namespace manual
}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
