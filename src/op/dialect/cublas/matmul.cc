/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas.h>
#include "raf/op.h"

#include "./cublas_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "../../../profiler/cuda/cuda_profiler.h"

namespace raf {
namespace op {
namespace cublas {
namespace manual {

using namespace raf::value;

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
class MatmulImpl : public raf::op::OpEnv {
  std::string env_name_;

 public:
  explicit MatmulImpl(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.matmul");
    static auto fschema_index = op::GetOpAttr<op::FRAFSchemaFieldIndex>(op, "FRAFSchemaFieldIndex");
    this->arg_indices = {
        fschema_index("x1"),
        fschema_index("x2"),
    };
    auto args = cv->args.as<op::schema::BinaryArgs>();
    CHECK(args != nullptr);
    std::string op_name = "raf.op.cublas.matmul";
    if (transpose_a || transpose_b) {
      op_name += "_";
      op_name += (transpose_a) ? "t" : "n";
      op_name += (transpose_b) ? "t" : "n";
    }
    env_name_ = TruncateName(GetUniqueName(op_name));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::BinaryArgs>();
    std::string op_name = tvm::runtime::Downcast<value::OpValue>(cv->callee)->op->name;
    GemmImpl(args->x1, transpose_a, args->x2, transpose_b, cv->out);
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

RAF_REGISTER_DIALECT_OP(cublas, matmul, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_nt, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_tn, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_tt, 15);
RAF_REGISTER_DIALECT_OP(cublas, dense, 15);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul", MatmulNN::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_nt", MatmulNT::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_tn", MatmulTN::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_tt", MatmulTT::make);
RAF_OP_ENV_MAKER("raf.op.cublas.dense", MatmulNT::make);

}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace raf
