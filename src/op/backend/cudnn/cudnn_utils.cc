/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/backend/cudnn/cudnn_utils.h
 * \brief Helper functions for cuDNN
 */
#include "dmlc/thread_local.h"
#include "./cudnn_utils.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {

CUDNNThreadEntry::CUDNNThreadEntry() {
  cudnnCreate(&handle);
}

using CUDNNThreadStore = dmlc::ThreadLocalStore<CUDNNThreadEntry>;

CUDNNThreadEntry* CUDNNThreadEntry::ThreadLocal() {
  return CUDNNThreadStore::Get();
}

}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
