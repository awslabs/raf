/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */

#include <cublas.h>
#include <stdint.h>
#include "mnm/op.h"

#include <numeric>
#include "mnm/device_api.h"
#include "../../schema/ufunc.h"
#include "../tvmjit/tvmjit_utils.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "./cublas_utils.h"
#include "../../../profiler/cuda/cuda_profiler.h"

namespace mnm {
namespace op {
namespace cublas {
namespace manual {

using namespace mnm::value;
using namespace mnm::device_api;
void GemmBatchedImpl(DLTensor* a, bool transpose_a, DLTensor* b, bool transpose_b, DLTensor* c) {
  Device dev(DevType::kCUDA(), 0);
  auto handle = CUBlasThreadEntry::ThreadLocal()->handle;

  cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  CHECK_EQ(a->ndim, 3U) << "Expected 3D tensor a, but got " << a->ndim;
  CHECK_EQ(b->ndim, 3U) << "Expected 3D tensor b, but got " << b->ndim;
  CHECK_EQ(c->ndim, 3U) << "Expected 3D tensor c, but got " << c->ndim;
  CHECK(a->shape[0] == c->shape[0] || b->shape[0] == c->shape[0])
      << "Batch size of tensor and output are mismatched";
  int m = c->shape[2];
  int n = c->shape[1];
  int k = b->shape[1 + (transb != CUBLAS_OP_N)];

  int batch_count = c->shape[0];
  int ldb = std::max(1, transpose_b ? k : m);
  int lda = std::max(1, transpose_a ? n : k);

  long long int strideA = n * k;
  long long int strideB = m * k;
  long long int strideC = m * n;
  // check broadcast
  if (a->shape[0] != b->shape[0]) {
    if (a->shape[0] == 1)
      strideA = 0;
    else if (b->shape[0] == 1)
      strideB = 0;
  }

  if (c->dtype.code == kDLFloat) {
    switch (c->dtype.bits) {
      case 16:
        CUBLAS_CALL(cublasHgemmStridedBatched(
            handle, transb, transa, m, n, k,
            static_cast<const __half*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<__half*>(b->data), ldb, strideB, static_cast<__half*>(a->data), lda,
            strideA, static_cast<const __half*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<__half*>(c->data), m, strideC, batch_count));
        return;
      case 32:
        CUBLAS_CALL(cublasSgemmStridedBatched(
            handle, transb, transa, m, n, k,
            static_cast<const float*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(b->data), ldb, strideB, static_cast<float*>(a->data), lda, strideA,
            static_cast<const float*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<float*>(c->data), m, strideC, batch_count));
        return;
      case 64:
        CUBLAS_CALL(cublasDgemmStridedBatched(
            handle, transb, transa, m, n, k,
            static_cast<const double*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
            static_cast<double*>(b->data), ldb, strideB, static_cast<double*>(a->data), lda,
            strideA, static_cast<const double*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
            static_cast<double*>(c->data), m, strideC, batch_count));
        return;
    }
  }

  CUBLAS_CALL(cublasGemmBatchedEx(
      handle, transb, transa, m, n, k, const_addr<1>(cudaDataType_t(DType(c->dtype))),
      static_cast<const void* const*>(b->data), cudaDataType_t(DType(b->dtype)), ldb,
      static_cast<const void* const*>(a->data), cudaDataType_t(DType(a->dtype)), lda,
      const_addr<0>(cudaDataType_t(DType(c->dtype))), static_cast<void* const*>(c->data),
      cudaDataType_t(DType(c->dtype)), m, batch_count, cudaDataType_t(DType(c->dtype)),
      CUBLAS_GEMM_DEFAULT));
}

template <bool transpose_a, bool transpose_b>
class BatchMatmulImpl : public mnm::op::OpEnv {
 public:
  explicit BatchMatmulImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    auto op = ir::Op::Get("mnm.op.batch_matmul");
    this->arg_indices = {
        fschema_index[op]("x1"),
        fschema_index[op]("x2"),
    };
    auto args = cv->args.as<op::schema::BinaryArgs>();
    CHECK(args != nullptr);
    std::string op_name = "mnm.op.batch_matmul";
    if (transpose_a || transpose_b) {
      op_name += "_";
      op_name += (transpose_a) ? "t" : "n";
      op_name += (transpose_b) ? "t" : "n";
    }
    env_name = tvmjit::TruncateName(tvmjit::GetUniqueName(op_name));
  }
  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::BinaryArgs>();
    std::string op_name = tvm::runtime::Downcast<value::OpValue>(cv->callee)->op->name;
    WITH_CUDA_PROFILER(cv->device, op_name, "ComputationOperator", {},
                       { GemmBatchedImpl(args->x1, transpose_a, args->x2, transpose_b, cv->out); })
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    DLTensor* x1 = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* x2 = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    GemmBatchedImpl(x1, transpose_a, x2, transpose_b, out);
  }
  static OpEnv* make(const CallValues& cv) {
    return new BatchMatmulImpl<transpose_a, transpose_b>(cv);
  }
};

using BatchMatmulNN = BatchMatmulImpl<false, false>;
using BatchMatmulNT = BatchMatmulImpl<false, true>;
using BatchMatmulTN = BatchMatmulImpl<true, false>;
using BatchMatmulTT = BatchMatmulImpl<true, true>;

MNM_OP_DISPATCH("mnm.op.batch_matmul", BatchMatmulNN::make, DevType::kCUDA(),
                "cublas");  // default setting, transposeA = false, transposeB = true
MNM_OP_DISPATCH("mnm.op.batch_matmul_nt", BatchMatmulNT::make, DevType::kCUDA(),
                "cublas");  // default setting, transposeA = false, transposeB = true
MNM_OP_DISPATCH("mnm.op.batch_matmul_tn", BatchMatmulTN::make, DevType::kCUDA(),
                "cublas");  // default setting, transposeA = false, transposeB = true
MNM_OP_DISPATCH("mnm.op.batch_matmul_tt", BatchMatmulTT::make, DevType::kCUDA(),
                "cublas");  // default setting, transposeA = false, transposeB = true
}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace mnm
