/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/backend/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas_v2.h>
#include <algorithm>
#include "dmlc/thread_local.h"
#include "./cublas_utils.h"

namespace mnm {
namespace op {
namespace backend {
namespace cublas {

using CUBlasThreadStore = dmlc::ThreadLocalStore<CUBlasThreadEntry>;

CUBlasThreadEntry::CUBlasThreadEntry() {
  CUBLAS_CALL(cublasCreate(&handle));
}

CUBlasThreadEntry* CUBlasThreadEntry::ThreadLocal() {
  return CUBlasThreadStore::Get();
}

}  // namespace cublas
}  // namespace backend
}  // namespace op
}  // namespace mnm