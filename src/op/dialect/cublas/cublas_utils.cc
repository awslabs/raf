/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas_v2.h>
#include <algorithm>
#include "dmlc/thread_local.h"
#include "./cublas_utils.h"

namespace raf {
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

RAF_REGISTER_DIALECT("cublas").set_enable(DevType::kCUDA());
TVM_REGISTER_PASS_CONFIG_OPTION("raf.cublas.allow_tf32", tvm::Bool);

}  // namespace cublas
}  // namespace op
}  // namespace raf
