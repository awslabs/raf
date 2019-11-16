/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/memory_pool.cc
 * \brief MNM memory pool manager
 */
#include <unordered_map>
#include "mnm/base.h"
#include "mnm/memory_pool.h"
#include "mnm/registry.h"

namespace mnm {
namespace memory_pool {

using registry::GetPackedFunc;
using registry::PerContextStore;

static std::unordered_map<int, std::string> default_strategies = {
    {DevType(DevType::kCPU()), "no_pool"},
    {DevType(DevType::kCUDA()), "no_pool"},
};

class MemoryPoolManager {
 public:
  static MemoryPoolManager* Get() {
    static MemoryPoolManager* instance = new MemoryPoolManager();
    return instance;
  }

  MemoryPool* GetPool(const Context& ctx, const std::string& name) {
    thread_local char maker_name[128];
    std::shared_ptr<MemoryPool>& result = reg.Get(ctx);
    if (result == nullptr) {
      std::lock_guard<std::mutex> lock(reg.mutex_);
      if (result == nullptr) {
        // ok, it is truly a nullptr
        if (name == "") {
          const std::string& default_name = default_strategies[ctx.device_type];
          snprintf(maker_name, sizeof(maker_name),
                   "mnm.memory_pool._make.%s", default_name.c_str());
        } else {
          snprintf(maker_name, sizeof(maker_name),
                   "mnm.memory_pool._make.%s", name.c_str());
        }
        void* ret = GetPackedFunc(maker_name)(ctx.operator DLContext());
        result.reset(static_cast<MemoryPool*>(ret));
        return result.get();
      }
    }
    // otherwise this is not nullptr
    CHECK_EQ(name, "");
    return result.get();
  }

  void Remove(const Context& ctx) {
    std::lock_guard<std::mutex> lock(reg.mutex_);
    std::shared_ptr<MemoryPool>& result = reg.Get(ctx);
    CHECK(result != nullptr);
    result = nullptr;
  }

 public:
  PerContextStore<MemoryPool, false> reg;
};

std::shared_ptr<Memory> Memory::Alloc(const Context& ctx, int64_t nbytes, int64_t alignment) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(ctx, "")->Alloc(nbytes, alignment);
}

std::vector<std::shared_ptr<Memory> > Memory::AllocMany(const Context& ctx,
                                                        const std::vector<int64_t>& nbytes,
                                                        int64_t alignment) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(ctx, "")->AllocMany(nbytes, alignment);
}

void Memory::RemovePool(const Context& ctx) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  mgr->Remove(ctx);
}

MemoryPool* Memory::GetPool(const Context& ctx) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(ctx, "");
}

MemoryPool* Memory::InitPool(const Context& ctx, const std::string& name) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(ctx, name);
}

}  // namespace memory_pool
}  // namespace mnm
