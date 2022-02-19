/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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

MNM_REGISTER_DIALECT("cublas").set_enable(DevType::kCUDA());
TVM_REGISTER_PASS_CONFIG_OPTION("mnm.cublas.allow_tf32", tvm::Bool);

}  // namespace cublas
}  // namespace op
}  // namespace mnm
