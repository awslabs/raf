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

using common::shape_utils::BytesCompactTensor;
using common::shape_utils::MakeShape;
using ir::Array;
using ir::Attrs;
using value::Value;

// TODO(@were): Is this good to put this as a global variable?

static AlgorithmCache<cudnnConvolutionFwdAlgo_t> _conv_fwd_alg_cache;

cudnnConvolutionFwdAlgo_t FindConvolutionForwardAlgorithm(const std::vector<int64_t>& key,
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

class Conv2DCUDNN : public mnm::op::OpEnv {
 public:
  DType dtype;
  Context ctx;
  const void* alpha;
  cudnnTensorDescriptor_t in_desc;
  cudnnFilterDescriptor_t ker_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t algo;
  size_t ws_size;
  void* ws;
  const void* beta;
  cudnnTensorDescriptor_t out_desc;

  Conv2DCUDNN(ir::Array<value::Value> args, const OpInfo& info, ir::Attrs attrs) {
    const auto dlts = common::arg_utils::AsVector(args);
    dtype = common::arg_utils::DeduceDLType(dlts);

    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));

    FORM_SHAPE(in_shape, dlts[0]);
    FORM_STRIDE(in_stride, in_shape);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc, CUDNNDType(dtype), in_shape.size(),
                                          dmlc::BeginPtr(in_shape), dmlc::BeginPtr(in_stride)));

    CUDNN_CALL(cudnnCreateFilterDescriptor(&ker_desc));
    FORM_SHAPE(ker_shape, dlts[1]);
    CUDNN_CALL(cudnnSetFilterNdDescriptor(ker_desc, CUDNNDType(dtype), CUDNN_TENSOR_NCHW,
                                          ker_shape.size(), dmlc::BeginPtr(ker_shape)));

    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    auto c2a = attrs.as<mnm::op::attrs::ConvAttrs>();
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(
        conv_desc, c2a->padding.size(), dmlc::BeginPtr(MakeShape<int>(c2a->padding)),
        dmlc::BeginPtr(MakeShape<int>(c2a->stride)), dmlc::BeginPtr(MakeShape<int>(c2a->dilation)),
        CUDNN_CROSS_CORRELATION, CUDNNDType(dtype)));

    const DLTensor* out = info->output;
    FORM_SHAPE(out_shape, out);
    FORM_STRIDE(out_stride, out_shape);
    int out_size = BytesCompactTensor(*out);
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc, CUDNNDType(dtype), out->ndim,
                                          dmlc::BeginPtr(out_shape), dmlc::BeginPtr(out_stride)));

    RequestMemory(const_cast<void**>(&out->data), info->ctx, out_size);

    std::vector<int64_t> key;
    VecAppend(key, in_shape);
    VecAppend(key, ker_shape);
    VecAppend(key, out_shape);

    algo = FindConvolutionForwardAlgorithm(key, in_desc, ker_desc, conv_desc, out_desc);
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(CUDNNThreadEntry::ThreadLocal()->handle,
                                                       in_desc, ker_desc, conv_desc, out_desc, algo,
                                                       &ws_size));
    RequestWorkspace(&ws, info->ctx, ws_size);

    alpha = CUDNNDType(dtype).const_addr<1>();
    beta = CUDNNDType(dtype).const_addr<0>();
  }

  ~Conv2DCUDNN() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(ker_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  }

  void Execute(Array<Value> args, const OpInfo& info, Attrs) override final {
    const DLTensor* in = args[0];
    const DLTensor* fil = args[1];
    const DLTensor* out = info->output;

    CUDNN_CALL(cudnnConvolutionForward(CUDNNThreadEntry::ThreadLocal()->handle, alpha, in_desc,
                                       in->data, ker_desc, fil->data, conv_desc, algo, ws, ws_size,
                                       beta, out_desc, out->data));
    CUDA_CALL(cudaDeviceSynchronize());
  }

  static OpEnv* make(ir::Array<value::Value> args, const OpInfo& info, ir::Attrs attrs) {
    std::unique_ptr<Conv2DCUDNN> res = std::make_unique<Conv2DCUDNN>(args, info, attrs);
    return res.release();
  }
};

// We now have generated version, this poc is not useful for now.
// MNM_REGISTER_OP_DISPATCH("mnm.op.conv2d", DevType::kCUDA(), "manual_cudnn", Conv2DCUDNN::make);

}  // namespace manual
}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
