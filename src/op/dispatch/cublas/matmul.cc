/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dispatch/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas.h>
#include "mnm/op.h"

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

void GemmImpl(DLTensor* a, bool transpose_a, DLTensor* b, bool transpose_b, DLTensor* c) {
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
      case 16: {
        CUBLAS_CALL(cublasGemmEx(handle, transb, transa, m, n, k, const_addr<1>(CUDA_R_32F),
                                 b->data, cudaDataType_t(DType(b->dtype)), ldb, a->data,
                                 cudaDataType_t(DType(a->dtype)), lda, const_addr<0>(CUDA_R_32F),
                                 c->data, cudaDataType_t(DType(c->dtype)), m, CUDA_R_32F,
                                 CUBLAS_GEMM_DFALT_TENSOR_OP));
        return;
      }
      case 32: {
        CUBLAS_CALL(
            cublasSgemm(handle, transb, transa, m, n, k,
                        static_cast<const float*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<float*>(b->data), ldb, static_cast<float*>(a->data), lda,
                        static_cast<const float*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<float*>(c->data), m));
        return;
      }
      case 64: {
        CUBLAS_CALL(
            cublasDgemm(handle, transb, transa, m, n, k,
                        static_cast<const double*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<double*>(b->data), ldb, static_cast<double*>(a->data), lda,
                        static_cast<const double*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<double*>(c->data), m));
        return;
      }
    }
  }
  CUBLAS_CALL(cublasGemmEx(
      handle, transb, transa, m, n, k, const_addr<1>(cudaDataType_t(DType(c->dtype))), b->data,
      cudaDataType_t(DType(b->dtype)), ldb, a->data, cudaDataType_t(DType(a->dtype)), lda,
      const_addr<0>(cudaDataType_t(DType(c->dtype))), c->data, cudaDataType_t(DType(c->dtype)), m,
      cudaDataType_t(DType(c->dtype)), CUBLAS_GEMM_DEFAULT));
}

template <bool transpose_a, bool transpose_b>
class MatmulImpl : public mnm::op::OpEnv {
 public:
  explicit MatmulImpl(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    auto op = ir::Op::Get("mnm.op.matmul");
    this->arg_indices = {
        fschema_index[op]("x1"),
        fschema_index[op]("x2"),
    };
    auto args = cv->args.as<op::schema::BinaryArgs>();
    CHECK(args != nullptr);
    std::string op_name = "mnm.op.matmul";
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
                       { GemmImpl(args->x1, transpose_a, args->x2, transpose_b, cv->out); })
  }
  void Execute(const std::vector<Value>& inputs, Value output) {
    DLTensor* x1 = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* x2 = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    GemmImpl(x1, transpose_a, x2, transpose_b, out);
  }
  static OpEnv* make(const CallValues& cv) {
    return new MatmulImpl<transpose_a, transpose_b>(cv);
  }
};

using MatmulNN = MatmulImpl<false, false>;
using MatmulNT = MatmulImpl<false, true>;
using MatmulTN = MatmulImpl<true, false>;
using MatmulTT = MatmulImpl<true, true>;

MNM_OP_DISPATCH("mnm.op.matmul", MatmulNN::make, DevType::kCUDA(), "cublas");
MNM_OP_DISPATCH("mnm.op.matmul_nt", MatmulNT::make, DevType::kCUDA(), "cublas");
MNM_OP_DISPATCH("mnm.op.matmul_tn", MatmulTN::make, DevType::kCUDA(), "cublas");
MNM_OP_DISPATCH("mnm.op.matmul_tt", MatmulTT::make, DevType::kCUDA(), "cublas");
MNM_OP_DISPATCH("mnm.op.dense", MatmulNT::make, DevType::kCUDA(), "cublas");

}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace mnm
