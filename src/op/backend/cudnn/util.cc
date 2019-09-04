#include <dmlc/thread_local.h>

#include "./util.h"

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

}
}
}
}
