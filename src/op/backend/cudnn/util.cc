#include <algorithm>

#include <dmlc/thread_local.h>

#include <mnm/value.h>

#include "./util.h"

namespace mnm {
namespace op {
namespace backend {
namespace cudnn {

using ir::Array;
using ir::Attrs;
using ir::Integer;
using value::Value;

using CUDNNThreadStore = dmlc::ThreadLocalStore<CUDNNThreadEntry>;

CUDNNThreadEntry::CUDNNThreadEntry() {
  CUDNN_CALL(cudnnCreate(&handle));
}

CUDNNThreadEntry* CUDNNThreadEntry::ThreadLocal() {
  return CUDNNThreadStore::Get();
}

}  // namespace cudnn
}  // namespace backend
}  // namespace op
}  // namespace mnm
