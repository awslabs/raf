#include <cublas.h>
#include <mnm/op.h>

#include "../../../common/arg_utils.h"
#include "../../../common/cuda.h"
#include "../../../common/shape_utils.h"
#include "./util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cublas {
namespace manual {

using common::shape_utils::MakeShape;
using ir::Array;
using ir::Attrs;
using value::Value;

class GemmCUBLAS : public mnm::op::OpEnv {
 public:
  DType dtype;
  Context ctx;

  GemmCUBLAS() {
  }

  void PreAllocate(ir::Array<value::Value> args, ir::Attrs attrs);
  void RequestMemory(void** dest, Context ctx, int64_t nb);
  void RequestWorkspace(void** dest, Context ctx, int64_t nb);

  void Execute(ir::Array<value::Value> args, ir::Attrs) override final;

  ~GemmCUBLAS() {
  }

  static OpEnv* make(ir::Array<value::Value> args, ir::Attrs attrs) {
    std::unique_ptr<GemmCUBLAS> res = std::make_unique<GemmCUBLAS>();
    res->PreAllocate(args, attrs);
    return res.release();
  }
};

using common::shape_utils::PadDims;
using common::shape_utils::Shape2Strides;

// To be compatible with Fortran, cublas uses column-major storage.
// However, C++ is row major. Thus, only compact-stored tensors can be
// fed to this executor.
void GemmCUBLAS::Execute(Array<Value> args, Attrs) {
  auto handle = CUBlasThreadEntry::ThreadLocal()->handle;
  const auto dlts = common::arg_utils::AsVector(args);

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;

  int m = dlts[1]->shape[0];
  int n = dlts[0]->ndim == 2 ? dlts[0]->shape[0] : 1;
  int k = dlts[0]->ndim == 2 ? dlts[0]->shape[1] : dlts[0]->shape[0];
  int lda = m;
  int ldb = k;
  int ldc = m;

  if (dtype.code == DTypeCode::kFloat()) {
    switch (dtype.bits) {
      case 32:
        CUBLAS_CALL(cublasSgemm(
            handle, transa, transb, m, n, k, (float*)common::cuda::const_addr<float, 1>(),
            (float*)dlts[1]->data, lda, (float*)dlts[0]->data, ldb,
            (float*)common::cuda::const_addr<float, 0>(), (float*)dlts[2]->data, ldc));
        break;
      case 64:
        CUBLAS_CALL(cublasDgemm(
            handle, transa, transb, m, n, k, (double*)common::cuda::const_addr<double, 1>(),
            (double*)dlts[0]->data, lda, (double*)dlts[1]->data, ldb,
            (double*)common::cuda::const_addr<double, 0>(), (double*)dlts[2]->data, ldc));
        break;
      default:
        LOG(FATAL) << "ValueError: Not supported data type!\n";
        throw;
    }
  } else {
    LOG(FATAL) << "ValueError: Not supported data type!\n";
    throw;
  }
}

void GemmCUBLAS::PreAllocate(Array<Value> args, Attrs attrs) {
  const auto dlts = common::arg_utils::AsVector(args);
  ctx = common::arg_utils::DeduceCtx(dlts);
  dtype = common::arg_utils::DeduceDLType(dlts);
  int size = dtype.bits;
  for (int i = 0; i < dlts[2]->ndim; ++i) {
    size *= dlts[2]->shape[i];
  }
  RequestMemory(const_cast<void**>(&dlts[2]->data), ctx, (size - 1) / 8 + 1);
}

void GemmCUBLAS::RequestMemory(void** dest, Context ctx, int64_t nb) {
  CUDA_CALL(cudaMalloc(dest, nb));
}

void GemmCUBLAS::RequestWorkspace(void** dest, Context ctx, int64_t nb) {
  CUDA_CALL(cudaMalloc(dest, nb));
}

MNM_REGISTER_OP_DISPATCH("mnm.op.linear", DevType::kCUDA(), "test_cublas", GemmCUBLAS::make);

}  // namespace manual
}  // namespace cublas
}  // namespace backend
}  // namespace op
}  // namespace mnm
