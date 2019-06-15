#include <dmlc/registry.h>

#include <mnm/op.h>
#include <mnm/types.h>

#include "./requests.h"

using mnm::executor::Executor;
using mnm::requests::Requests;
using mnm::types::Context;

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mnm::op::OpBackend);
DMLC_REGISTRY_ENABLE(::mnm::op::OpDispatch);
}  // namespace dmlc

namespace mnm {
namespace op {

OpBackend::TDeviceRegistry* OpBackend::DeviceRegistry() {
  static OpBackend::TDeviceRegistry* registry = new OpBackend::TDeviceRegistry();
  return registry;
}

OpBackend* OpBackend::Get(const std::string& name) {
  return &Registry()->__REGISTER_OR_GET__(name);
}

OpDispatch::TDispatchList& OpDispatch::Get(const std::string& op_name,
                                           mnm::types::DeviceType device_type) {
  OpDispatch& op_dispatch = Registry()->__REGISTER_OR_GET__(op_name);
  TDispatchList* ret = nullptr;
  op_dispatch.dispatch.Read(device_type, [&ret](TDispatchList* list) { ret = list; });
  return *ret;
}

OpBackend::TRegistry* OpBackend::Registry() {
  return TRegistry::Get();
}

OpDispatch::TRegistry* OpDispatch::Registry() {
  return TRegistry::Get();
}

class OpEnv::Impl {
 public:
  friend class OpEnv;

  Impl() : requests(new Requests()) {
  }

  ~Impl() = default;

  void RequestMemory(void** dest, Context ctx, int64_t nbytes) {
    if (executor == nullptr) {
      requests->memory.push_back({dest, ctx, nbytes});
    } else {
      // TODO(@junrushao1994): eagerly call from executor
    }
  }

  void RequestWorkspace(void** dest, Context ctx, int64_t nbytes) {
    if (executor == nullptr) {
      requests->workspace.push_back({dest, ctx, nbytes});
    } else {
      // TODO(@junrushao1994): eagerly call from executor
    }
  }

  void RequestStream(void** dest, Context ctx) {
    if (executor == nullptr) {
      requests->stream.push_back({dest, ctx});
    } else {
      // TODO(@junrushao1994): eagerly call from executor
    }
  }

  void RequestDistributed(void** dest) {
    if (executor == nullptr) {
      requests->dist.push_back({dest});
    } else {
      // TODO(@junrushao1994): eagerly call from executor
    }
  }

  Requests* SetExecutor(Executor* exec) {
    CHECK(this->executor == nullptr);
    this->executor = exec;
    return requests.release();
  }

 public:
  Executor* executor = nullptr;
  std::unique_ptr<Requests> requests;
};

OpEnv::OpEnv() : impl(new OpEnv::Impl()) {
}

OpEnv::~OpEnv() = default;

void OpEnv::RequestMemory(void** dest, Context ctx, int64_t nbytes) {
  impl->RequestMemory(dest, ctx, nbytes);
}

void OpEnv::RequestWorkspace(void** dest, Context ctx, int64_t nbytes) {
  impl->RequestWorkspace(dest, ctx, nbytes);
}

void OpEnv::RequestStream(void** dest, Context ctx) {
  impl->RequestStream(dest, ctx);
}

void OpEnv::RequestDistributed(void** dest) {
  impl->RequestDistributed(dest);
}

void* OpEnv::SetExecutor(Executor* executor) {
  return impl->SetExecutor(executor);
}

}  // namespace op
}  // namespace mnm
