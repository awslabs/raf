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
 * \file src/op/dialect/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#include "dmlc/thread_local.h"
#include "./cudnn_utils.h"

namespace mnm {
namespace op {
namespace cudnn {

CUDNNThreadEntry::CUDNNThreadEntry() {
  CUDNN_CALL(cudnnCreate(&handle));
}

using CUDNNThreadStore = dmlc::ThreadLocalStore<CUDNNThreadEntry>;

CUDNNThreadEntry* CUDNNThreadEntry::ThreadLocal() {
  return CUDNNThreadStore::Get();
}

bool CudnnConfigGetBenchmark() {
  return CUDNNThreadEntry::ThreadLocal()->benchmark;
}

void CudnnConfigSetBenchmark(bool benchmark) {
  CUDNNThreadEntry::ThreadLocal()->benchmark = benchmark;
}

MNM_REGISTER_DIALECT("cudnn").set_enable(DevType::kCUDA());
MNM_REGISTER_GLOBAL("mnm.backend.cudnn.ConfigGetBenchmark").set_body_typed(CudnnConfigGetBenchmark);
MNM_REGISTER_GLOBAL("mnm.backend.cudnn.ConfigSetBenchmark").set_body_typed(CudnnConfigSetBenchmark);

}  // namespace cudnn
}  // namespace op
}  // namespace mnm
