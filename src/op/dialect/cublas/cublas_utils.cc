/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dialect/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas_v2.h>
#include <algorithm>
#include "dmlc/thread_local.h"
#include "./cublas_utils.h"

namespace mnm {
namespace op {
namespace cublas {

using CUBlasThreadStore = dmlc::ThreadLocalStore<CUBlasThreadEntry>;

CUBlasThreadEntry::CUBlasThreadEntry() {
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLASTryEnableTensorCore(handle);
}

CUBlasThreadEntry* CUBlasThreadEntry::ThreadLocal() {
  return CUBlasThreadStore::Get();
}

TVM_REGISTER_PASS_CONFIG_OPTION("mnm.cublas.allow_tf32", tvm::Bool);

}  // namespace cublas
}  // namespace op
}  // namespace mnm
