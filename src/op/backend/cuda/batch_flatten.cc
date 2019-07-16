#include <cuda.h>
#include <mnm/op.h>

#include "../../../common/arg_utils.h"
#include "../../../common/cuda.h"

namespace mnm {
namespace op {
namespace backend {
namespace cuda {

using common::arg_utils::AsVector;
using common::arg_utils::DeduceCtx;
using common::arg_utils::DeduceDLType;

class BatchFlattenCUDA : public mnm::op::OpEnv {
 public:
  DType dtype;
  Context ctx;
  int64_t size;

  BatchFlattenCUDA() {
  }

  void PreAllocate(rly::Array<value::Value> args, rly::Attrs attrs) {
    auto dlts = AsVector(args);
    dtype = DeduceDLType(dlts);
    ctx = DeduceCtx(dlts);
    size = (dlts[1]->shape[0] * dlts[1]->shape[1] * dtype.bits - 1) / 8 + 1;
    RequestMemory(const_cast<void**>(&dlts[1]->data), ctx, size);
  }

  void Execute(rly::Array<value::Value> args, rly::Attrs attrs) override {
    auto dlts = AsVector(args);
    CUDA_CALL(cudaMemcpy(dlts[1]->data, dlts[0]->data, size, cudaMemcpyDeviceToDevice));
  }

  static mnm::op::OpEnv* make(rly::Array<value::Value> args, rly::Attrs attrs) {
    std::unique_ptr<BatchFlattenCUDA> res = std::make_unique<BatchFlattenCUDA>();
    res->PreAllocate(args, attrs);
    return res.release();
  }

  // TODO(@were): Delete this when executor is implemented.
  // TODO(@junrushao1994): Implement the executor.
  void RequestMemory(void** dest, Context ctx, int64_t nb) {
    CUDA_CALL(cudaMalloc(dest, nb));
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.batch_flatten", DevType::kCUDA(), "cuda_sync",
                         BatchFlattenCUDA::make);

}  // namespace cuda
}  // namespace backend
}  // namespace op
}  // namespace mnm
