/*!
 * Copyright (c) 2019 by Contributors
 * \file src/memory_pool/no_pool/no_pool.cc
 * \brief No memory pool
 */
#include <atomic>
#include "mnm/device_api.h"
#include "mnm/memory_pool.h"
#include "mnm/registry.h"

namespace mnm {
namespace memory_pool {
namespace no_pool {

using device_api::DeviceAPI;

class NonOwnedMemory final : public Memory {
 public:
  explicit NonOwnedMemory(void* data, const Context& ctx, std::shared_ptr<DeviceAPI> api) {
    this->data = data;
    this->ctx = ctx;
    this->api = std::move(api);
  }

  ~NonOwnedMemory() {
    if (data != nullptr) {
      api->FreeMemory(data);
    }
  }

 public:
  std::shared_ptr<DeviceAPI> api;
};

class NoPool final : public MemoryPool {
 public:
  explicit NoPool(Context ctx) {
    this->ctx = ctx;
    this->api = DeviceAPI::Get(ctx.device_type);
  }

  std::shared_ptr<Memory> Alloc(int64_t nbytes, int64_t alignment) override {
    CHECK_GE(nbytes, 0);
    void* data = nullptr;
    if (nbytes > 0) {
      data = api->AllocMemory(nbytes, alignment);
    }
    return std::make_shared<NonOwnedMemory>(data, ctx, api);
  }

  std::vector<std::shared_ptr<Memory>> AllocBatch(const std::vector<int64_t>& nbytes,
                                                  int64_t alignment) override {
    std::vector<std::shared_ptr<Memory>> ret;
    ret.reserve(nbytes.size());
    for (int64_t bytes : nbytes) {
      ret.emplace_back(Alloc(bytes, alignment));
    }
    return ret;
  }

 public:
  static void* make(DLContext ctx) {
    return new NoPool(ctx);
  }

  Context ctx;
  std::shared_ptr<DeviceAPI> api;
};

MNM_REGISTER_GLOBAL("mnm.memory_pool._make.no_pool").set_body_typed(NoPool::make);

}  // namespace no_pool
}  // namespace memory_pool
}  // namespace mnm
