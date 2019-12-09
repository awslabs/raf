/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/backend/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */

#include <cublas.h>
#include <mnm/op.h>

#include "../../schema/gemm.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "./cublas_utils.h"

namespace mnm {
namespace op {
namespace backend {
namespace cublas {
namespace manual {

using value::Value;

class CUBLASMatmul : public mnm::op::OpEnv {
 public:
  explicit CUBLASMatmul(const CallValues& cv) {
    auto args = cv->args.as<schema::MatmulArgs>();
    CHECK(args != nullptr);
    DLTensor* out = cv->out;
    RequestMemory(&out->data, cv->ctx, common::shape_utils::BytesCompactTensor(*out));
  }

  void Execute(const CallValues& cv) override;

  ~CUBLASMatmul() {
  }

  static OpEnv* make(const CallValues& cv) {
    return new CUBLASMatmul(cv);
  }
};

// To be compatible with Fortran, cublas uses column-major storage.
// However, C++ is row major. Thus, only compact-stored tensors can be
// fed to this executor.
void CUBLASMatmul::Execute(const CallValues& cv) {
  auto handle = CUBlasThreadEntry::ThreadLocal()->handle;
  auto args = cv->args.as<schema::MatmulArgs>();

  cublasOperation_t transa = args->transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = args->transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  DLTensor* a = args->a;
  DLTensor* b = args->b;
  DLTensor* c = cv->out;

  int m = c->shape[1];
  int n = c->shape[0];
  int k = b->shape[transb != CUBLAS_OP_N];

  int ldb = std::max(1, args->transpose_b ? k : m);
  int lda = std::max(1, args->transpose_a ? n : k);

  if (c->dtype.code == kDLFloat) {
    switch (c->dtype.bits) {
      case 32:
        CUBLAS_CALL(cublasSgemm(
            handle, transb, transa, m, n, k,
            static_cast<const float*>(const_typed_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(b->data), ldb,
            static_cast<float*>(a->data), lda,
            static_cast<const float*>(const_typed_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(c->data), m));
        return;
      case 64:
        CUBLAS_CALL(cublasDgemm(
            handle, transb, transa, m, n, k,
            static_cast<const double*>(const_typed_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<double*>(b->data), ldb,
            static_cast<double*>(a->data), lda,
            static_cast<const double*>(const_typed_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<double*>(c->data), m));
        return;
    }
  }
  CUBLAS_CALL(cublasGemmEx(
      handle, transb, transa, m, n, k, const_typed_addr<1>(cudaDataType_t(DType(c->dtype))),
      b->data, cudaDataType_t(DType(b->dtype)), ldb, a->data, cudaDataType_t(DType(a->dtype)), lda,
      const_typed_addr<0>(cudaDataType_t(DType(c->dtype))), c->data,
      cudaDataType_t(DType(c->dtype)), m, cudaDataType_t(DType(c->dtype)), CUBLAS_GEMM_DEFAULT));
}

MNM_OP_DISPATCH("mnm.op.matmul", CUBLASMatmul, DevType::kCUDA(), "cublas");

}  // namespace manual
}  // namespace cublas
}  // namespace backend
}  // namespace op
}  // namespace mnm
