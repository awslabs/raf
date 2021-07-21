/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/memory_pool.cc
 * \brief MNM memory pool manager
 */
#include <unordered_map>
#include "mnm/base.h"
#include "mnm/memory_pool.h"
#include "mnm/registry.h"

#ifdef MNM_USE_CUDA
#include <cuda.h>
#else
#define CUDA_VERSION 0
#endif

namespace mnm {
namespace memory_pool {

using registry::GetPackedFunc;
using registry::PerDeviceStore;

static std::unordered_map<int, std::string> default_strategies = {
    {DevType(DevType::kCPU()), "page_unit_pool"},
#if CUDA_VERSION >= 11030
    {DevType(DevType::kCUDA()), "no_pool"}
#else
    {DevType(DevType::kCUDA()), "page_unit_pool"}
#endif
};

class MemoryPoolManager {
 public:
  static MemoryPoolManager* Get() {
    static MemoryPoolManager* instance = new MemoryPoolManager();
    return instance;
  }

  MemoryPool* GetPool(const Device& dev, const std::string& name) {
    thread_local char maker_name[128];
    std::shared_ptr<MemoryPool>& result = reg.Get(dev);
    if (result == nullptr) {
      std::lock_guard<std::mutex> lock(reg.mutex_);
      if (result == nullptr) {
        // ok, it is truly a nullptr
        pool_name = name;
        if (name == "") {
          pool_name = default_strategies[dev.device_type];
        }
        snprintf(maker_name, sizeof(maker_name), "mnm.memory_pool._make.%s", pool_name.c_str());
        void* ret = GetPackedFunc(maker_name)(dev.operator DLDevice());
        result.reset(static_cast<MemoryPool*>(ret));
        return result.get();
      }
    }
    // otherwise this is not nullptr
    return result.get();
  }

  void Remove(const Device& dev) {
    std::lock_guard<std::mutex> lock(reg.mutex_);
    std::shared_ptr<MemoryPool>& result = reg.Get(dev);
    result = nullptr;
  }

 public:
  PerDeviceStore<MemoryPool, false> reg;
  std::string pool_name = "";
};

int64_t Memory::GetAllocBytes(const Device& dev, int64_t nbytes) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, "")->GetAllocBytes(nbytes);
}

std::shared_ptr<Memory> Memory::Alloc(const Device& dev, int64_t nbytes, int64_t alignment) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, "")->Alloc(nbytes, alignment);
}

std::vector<std::shared_ptr<Memory> > Memory::AllocBatch(const Device& dev,
                                                         const std::vector<int64_t>& nbytes,
                                                         int64_t alignment) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, "")->AllocBatch(nbytes, alignment);
}

std::pair<float, float> Memory::GetPoolSize(const Device& dev) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, "")->GetPoolSize();
}

void Memory::RemovePool(const Device& dev) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  mgr->Remove(dev);
}

MemoryPool* Memory::ResetPool(const Device& dev) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  std::string pool_name = mgr->pool_name;
  mgr->Remove(dev);
  return mgr->GetPool(dev, pool_name);
}

MemoryPool* Memory::GetPool(const Device& dev) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, "");
}

MemoryPool* Memory::InitPool(const Device& dev, const std::string& name) {
  MemoryPoolManager* mgr = MemoryPoolManager::Get();
  return mgr->GetPool(dev, name);
}

/*!
 * \brief RemovePool Disable the current memory pool, the memory chuncks in this pool will not
 * be freed unitl there is nobody using to it.
 *
 * \param dev The device that the pool belongs to.
 */
void RemovePool(const Device& dev) {
  Memory::RemovePool(dev);
}

/*!
 * \brief ResetPool Enable a new memory pool with the same type as the current pool.
 *
 * \param dev The device that the pool belongs to.
 */
void ResetPool(const Device& dev) {
  Memory::ResetPool(dev);
}

/*!
 * \brief InitPool Enable a new memory pool using the given pool_name. The memories requested after
 * this will be managed by this pool.
 *
 * \param dev The device that the pool belongs to.
 * \param pool_name The name of the new pool.
 */
void InitPool(const Device& dev, std::string pool_name) {
  Memory::RemovePool(dev);  // Remove the current pool firstly.
  Memory::InitPool(dev, pool_name);
}

MNM_REGISTER_GLOBAL("mnm.memory_pool.InitPool")
    .set_body_typed([](const tvm::Device& dev, const std::string pool_name) {
      return InitPool(Device(dev), pool_name);
    });

MNM_REGISTER_GLOBAL("mnm.memory_pool.RemovePool").set_body_typed([](const tvm::Device& dev) {
  return RemovePool(Device(dev));
});

MNM_REGISTER_GLOBAL("mnm.memory_pool.ResetPool").set_body_typed([](const tvm::Device& dev) {
  return ResetPool(Device(dev));
});

}  // namespace memory_pool
}  // namespace mnm
