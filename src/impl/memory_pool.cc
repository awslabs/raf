#include <unordered_map>

#include <mnm/base.h>
#include <mnm/memory_pool.h>
#include <mnm/registry.h>

namespace mnm {
namespace memory_pool {

using registry::PerContextStore;
using registry::Registry;

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

  static MemoryPool* CreateMemoryPool(Context ctx, const std::string& name) {
    thread_local char creator_name[128];
    sprintf(creator_name, "mnm.memory_pool._make.%s", name.c_str());
    auto creator = Registry::Get(creator_name);
    CHECK(creator != nullptr);
    void* ret = (*creator)(ctx.operator DLContext());
    return static_cast<MemoryPool*>(ret);
  }

 public:
  PerContextStore<MemoryPool, false> reg;
};

std::shared_ptr<MemoryPool> MemoryPool::Get(Context ctx) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  std::shared_ptr<MemoryPool>& result = mgr->reg.Get(ctx);
  if (result == nullptr) {
    std::unique_lock<std::mutex> lock(mgr->reg.GrabLock());
    if (result == nullptr) {
      result.reset(MemoryPoolManager::CreateMemoryPool(ctx, default_strategies[ctx.device_type]));
    }
  }
  return result;
}

}  // namespace memory_pool
}  // namespace mnm
