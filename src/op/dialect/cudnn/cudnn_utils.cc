/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/dialect/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#include "dmlc/thread_local.h"
#include "./cudnn_utils.h"

namespace mnm {
namespace op {
namespace cudnn {

CUDNNThreadEntry::CUDNNThreadEntry() {
  cudnnCreate(&handle);
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
