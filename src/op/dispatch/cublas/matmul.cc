/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas.h>
#include "mnm/op.h"

#include "../../schema/gemm.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "./cublas_utils.h"

namespace mnm {
namespace op {
namespace cublas {
namespace manual {

using value::Value;

void GemmImpl(DLTensor *a, bool transpose_a, DLTensor *b, bool transpose_b, DLTensor *c) {
  auto handle = CUBlasThreadEntry::ThreadLocal()->handle;

  cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  int m = c->shape[1];
  int n = c->shape[0];
  int k = b->shape[transb != CUBLAS_OP_N];

  int ldb = std::max(1, transpose_b ? k : m);
  int lda = std::max(1, transpose_a ? n : k);

  if (c->dtype.code == kDLFloat) {
    switch (c->dtype.bits) {
      case 32:
        CUBLAS_CALL(cublasSgemm(
            handle, transb, transa, m, n, k,
            static_cast<const float*>(const_typed_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(b->data), ldb, static_cast<float*>(a->data), lda,
            static_cast<const float*>(const_typed_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(c->data), m));
        return;
      case 64:
        CUBLAS_CALL(cublasDgemm(
            handle, transb, transa, m, n, k,
            static_cast<const double*>(const_typed_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<double*>(b->data), ldb, static_cast<double*>(a->data), lda,
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

template<typename ArgT_, typename DerivedT>
class MatmulBase : public mnm::op::OpEnv {
 protected:
  using ArgType = ArgT_;
  void Init(const CallValues& cv) {
    auto args = cv->args.as<ArgType>();
    CHECK(args != nullptr);
    DLTensor* out = cv->out;
    RequestMemory(&out->data, cv->ctx, common::shape_utils::BytesCompactTensor(*out));
  }
 public:
  static OpEnv* make(const CallValues& cv) {
    return new DerivedT(cv);
  }
};

class Matmul : public MatmulBase<schema::MatmulArgs, Matmul> {
 public:
  explicit Matmul(const CallValues &cv) {
    Init(cv);
  }
  void Execute(const CallValues &cv) override {
    auto args = cv->args.as<Matmul::ArgType>();
    GemmImpl(args->a, args->transpose_a, args->b, args->transpose_b, cv->out);
  }
};

MNM_OP_DISPATCH("mnm.op.matmul", Matmul::make, DevType::kCUDA(), "cublas");

class MatmulDa : public MatmulBase<schema::MatmulDabArgs, MatmulDa> {
 public:
  explicit MatmulDa(const CallValues &cv) {
    Init(cv);
  }
  void Execute(const CallValues &cv) {
    auto args = cv->args.as<MatmulDa::ArgType>();
    if (!args->transpose_dy) {
      GemmImpl(args->dy, false, args->a_or_b, !args->transpose_dx, cv->out);
    } else {
      GemmImpl(args->a_or_b, args->transpose_dx, args->dy, true, cv->out);
    }
  }
};

MNM_OP_DISPATCH("mnm.op.matmul_da", MatmulDa::make, DevType::kCUDA(), "cublas");

class MatmulDb : public MatmulBase<schema::MatmulDabArgs, MatmulDb> {
 public:
  explicit MatmulDb(const CallValues &cv) {
    Init(cv);
  }
  void Execute(const CallValues &cv) {
    auto args = cv->args.as<MatmulDb::ArgType>();
    if (!args->transpose_dy) {
      GemmImpl(args->a_or_b, !args->transpose_dx, args->dy, false, cv->out);
    } else {
      GemmImpl(args->dy, true, args->a_or_b, args->transpose_dx, cv->out);
    }
  }
};

MNM_OP_DISPATCH("mnm.op.matmul_db", MatmulDb::make, DevType::kCUDA(), "cublas");

}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace mnm
