#pragma once

#include <mnm/device_api.h>
#include <mnm/types.h>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace mnm {
namespace memory_pool {

class MemoryPoolManager;

class MemoryChunk {
 public:
  void* data;
};

class MemoryPool {
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
   * TODO(@junrushao1994): replace size_t with dim_t.
   *
   * TODO(@junrushao1994): thread safety will be an issue with ReplacePool
   *
   * Note: the pool is designed to have no ownership to DeviceAPI. Therefore, please use with
   * caution if it is not managed by MemoryPoolManager.
   */
  friend MemoryPoolManager;

 protected:
  using DataType = mnm::types::DataType;
  using Context = mnm::types::Context;
  using FAlloc = std::function<void*(size_t nbytes, size_t alignment, DataType type_hint)>;
  using FDealloc = std::function<void(void* ReplacePool)>;

 public:
  virtual ~MemoryPool() = default;
  virtual MemoryChunk* Alloc(size_t nbytes, size_t alignment, DataType type_hint) = 0;
  virtual void Dealloc(MemoryChunk* mem) = 0;
  virtual void DeallocAll() = 0;

 private:
  static MemoryPool* Create(const char* name);
  void Init(Context ctx, bool allow_missing);

 protected:
  Context ctx_;
  FAlloc f_alloc_{nullptr};
  FDealloc f_dealloc_{nullptr};
};

class MemoryPoolManager final {
  using DataType = mnm::types::DataType;
  using DeviceAPIManager = mnm::device_api::DeviceAPIManager;
  using Context = mnm::types::Context;
  using DeviceType = mnm::types::DeviceType;
  using PoolPtr = std::unique_ptr<MemoryPool>;
  static constexpr int kMaxDeviceAPI = DeviceAPIManager::kMaxDeviceAPI;

 public:
  MemoryPoolManager() = default;
  ~MemoryPoolManager();
  MemoryPool* ReplacePool(Context ctx, const char* name, bool allow_missing);

  static std::shared_ptr<MemoryPoolManager> Global() {
    static std::shared_ptr<MemoryPoolManager> inst = std::make_shared<MemoryPoolManager>();
    return inst;
  }

  inline MemoryChunk* Alloc(Context ctx, size_t nbytes, size_t alignment, DataType type_hint) {
    return GetPoolPtr(ctx, nullptr, false, true)->Alloc(nbytes, alignment, type_hint);
  }

  inline void Dealloc(Context ctx, MemoryChunk* mem) {
    return GetPoolPtr(ctx, nullptr, false, true)->Dealloc(mem);
  }

 private:
  PoolPtr& GetPoolPtr(Context ctx, const char* name, bool allow_missing, bool create_if_missing);

 private:
  std::array<std::vector<PoolPtr>, kMaxDeviceAPI> pools_;
  std::mutex mutex_;

 private:
  std::shared_ptr<DeviceAPIManager> device_api_manager_{DeviceAPIManager::Global()};
};  // namespace memory_pool

}  // namespace memory_pool
}  // namespace mnm
