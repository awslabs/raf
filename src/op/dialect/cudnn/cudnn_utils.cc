/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#include "dmlc/thread_local.h"
#include "./cudnn_utils.h"

namespace raf {
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

RAF_REGISTER_DIALECT("cudnn").set_enable(DevType::kCUDA());
RAF_REGISTER_GLOBAL("raf.backend.cudnn.ConfigGetBenchmark").set_body_typed(CudnnConfigGetBenchmark);
RAF_REGISTER_GLOBAL("raf.backend.cudnn.ConfigSetBenchmark").set_body_typed(CudnnConfigSetBenchmark);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
