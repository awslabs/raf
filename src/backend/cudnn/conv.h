#include <cudnn.h>

#include <mnm/op.h>

#include "util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {
namespace manual {

class Conv2dCUDNN : public mnm::op::OpEnv {
 public:

  cudnnHandle_t handle;
  const void *alpha;
  cudnnTensorDescriptor_t in_desc;
  cudnnFilterDescriptor_t ker_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t algo;
  size_t ws_size;
  void *ws;
  const void *beta;
  cudnnTensorDescriptor_t out_desc;
  void *out_data;

  Conv2dCUDNN() {}
  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs);
  void RequestMemory(void **dest, Context ctx, int64_t nb);
  void RequestWorkspace(void **dest, Context ctx, int64_t nb);

  void Execute(rly::Array<value::Value> args, rly::Attrs) override final; 

  ~Conv2dCUDNN() {
    cudnnDestroy(handle);
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

} // namespace manual
} // namespace cudnn
} // namespace backend
} // namespace op
} // namespace mnm