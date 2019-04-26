#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <mnm/device_api.h>
#include <mnm/types.h>

namespace mnm {
namespace memory_pool {
/*
 * A memory pool is a high-level description of memory allocation strategy, along with the data
 * structure to store the data pointers. The strategy is designed to be independent of the
 * concrete device.
 *
 * We have the following assumption:
 *
 * 1) On one device there is only one memory pool, which is managed by the MemoryPoolManager
 * singleton.
 *
 * 2) One memory pool is only in charge of one device.
 *
 * 3) A memory pool could be released and replaced with another memory pool, as long as all of its
 * managed memory is released.
 *
 * TODO(@junrushao1994): replace size_t with index_t.
 *
 * TODO(@junrushao1994): thread safety will be an issue with ReplacePool
 *
 * Note: the pool is designed to have no ownership to DeviceAPI. Therefore, please use with
 * caution if it is not managed by MemoryPoolManager.
 */

class MemoryPool;
struct MemoryChunk {
  void* data;
};

class MemoryPoolManager final {
  // TODO(@junrushao1994): put it to a better place
  static constexpr int kMaxDeviceAPI = mnm::device_api::DeviceAPIManager::kMaxDeviceAPI;
  // Manager has exclusive ownership over the pools.
  using PoolPtr = std::unique_ptr<MemoryPool>;

 public:
  class Impl;
  friend class Impl;

 public:
  // default constructor
  MemoryPoolManager() = default;
  // deallocator specifies the destruction order
  ~MemoryPoolManager();
  // Return a memory chunk without any ownership
  MemoryChunk* Alloc(mnm::types::Context ctx, size_t nbytes, size_t alignment,
                     mnm::types::DType type_hint);
  // Find the correct memory pool, and call the dealloc method of the pool
  void Dealloc(mnm::types::Context ctx, MemoryChunk* mem);
  // Replace the memory pool with one with the given name
  MemoryPool* Replace(mnm::types::Context ctx, const char* name);

 public:
  // Global singleton
  static std::shared_ptr<MemoryPoolManager> Global() {
    static std::shared_ptr<MemoryPoolManager> inst = std::make_shared<MemoryPoolManager>();
    return inst;
  }

 private:
  std::array<std::vector<PoolPtr>, kMaxDeviceAPI> pools_;
  std::mutex mutex_;

 private:
  std::shared_ptr<mnm::device_api::DeviceAPIManager> device_api_manager_{
      mnm::device_api::DeviceAPIManager::Global()};
};

class MemoryPool {
  friend MemoryPoolManager;
  friend MemoryPoolManager::Impl;

 protected:
  // TODO(@junrushao1994): try use function pointers / structs if possible
  using FAlloc = std::function<void*(size_t, size_t, mnm::types::DType)>;
  using FDealloc = std::function<void(void*)>;

 public:
  MemoryPool() = default;
  virtual ~MemoryPool() = default;
  virtual MemoryChunk* Alloc(size_t nbytes, size_t alignment, mnm::types::DType type_hint) = 0;
  virtual void Dealloc(MemoryChunk* mem) = 0;
  virtual void DeallocAll() = 0;

 private:
  static MemoryPool* Create(const char* name);

 protected:
  mnm::types::Context ctx_hint_;
  FAlloc f_alloc_{nullptr};
  FDealloc f_dealloc_{nullptr};
};

}  // namespace memory_pool
}  // namespace mnm
