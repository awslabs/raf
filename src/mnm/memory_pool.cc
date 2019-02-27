#include <mnm/commons.h>
#include <mnm/memory_pool.h>
#include <mnm/registry.h>

namespace mnm {
namespace memory_pool {

using Registry = mnm::registry::Registry;
using DeviceAPI = mnm::device_api::DeviceAPI;
using DeviceAPIManager = mnm::device_api::DeviceAPIManager;
using Context = mnm::types::Context;
using DeviceType = mnm::types::DeviceType;
using DataType = mnm::types::DataType;
using PoolPtr = std::unique_ptr<MemoryPool>;

MemoryPool* MemoryPool::Create(const char* name) {
  static const std::string prefix("mnm.memory_pool.");
  std::string creator_name = prefix + name;
  auto creator = Registry::Get(creator_name);
  CHECK(creator != nullptr) << "ValueError: MemoryPool " << creator_name << " is not enabled.";
  void* ret = (*creator)();
  return static_cast<MemoryPool*>(ret);
}

void MemoryPool::Init(Context ctx, bool allow_missing) {
  DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(ctx.device_type, allow_missing);
  int device_id = ctx.device_id;
  this->ctx_ = ctx;
  this->f_alloc_ = (api == nullptr)
                       ? FAlloc(nullptr)
                       : [api, device_id](size_t nbytes, size_t alignment, DataType type_hint) {
                           return api->AllocMemory(device_id, nbytes, alignment, type_hint);
                         };
  this->f_dealloc_ = (api == nullptr) ? FDealloc(nullptr) : [api, device_id](void* ptr) {
    api->DeallocMemory(device_id, ptr);
  };
}

MemoryPoolManager::~MemoryPoolManager() {
  for (std::vector<PoolPtr>& pools : pools_) {
    for (PoolPtr& pool : pools) {
      pool->DeallocAll();
      pool = nullptr;
    }
  }
  device_api_manager_ = nullptr;
}

MemoryPool* MemoryPoolManager::ReplacePool(Context ctx, const char* name, bool allow_missing) {
  PoolPtr& ptr = GetPoolPtr(ctx, name, allow_missing, false);
  std::lock_guard<std::mutex> lock(mutex_);
  if (ptr != nullptr) {
    ptr->DeallocAll();
    ptr = nullptr;
  }
  ptr.reset(MemoryPool::Create(name));
  return ptr.get();
}

inline const char* GetDefaultPool(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::kDLCPU:
      return "no_pool";
    default:
      LOG(FATAL) << "InternalError: Default memory pool is not defined for " << device_type;
  }
  return nullptr;
}

inline PoolPtr& MemoryPoolManager::GetPoolPtr(Context ctx, const char* name, bool allow_missing,
                                              bool create_if_missing) {
  int device_type = int(ctx.device_type);
  int device_id = ctx.device_id;
  std::vector<PoolPtr>& pool_vec = pools_[device_type];
  LOCKED_IF(pool_vec.empty(), mutex_, {
    DeviceAPI* api = device_api_manager_->GetAPI(ctx.device_type, allow_missing);
    int n_devices = (api == nullptr) ? 0 : api->GetNDevices();
    CHECK_LT(device_id, n_devices) << "ValueError: Device " << device_id << " not found.";
    pool_vec.resize(n_devices);
  });
  int n_devices = pool_vec.size();
  CHECK_LT(device_id, n_devices) << "ValueError: Device " << device_id << " not found.";
  PoolPtr& ptr = pool_vec[device_id];
  LOCKED_IF(create_if_missing && ptr == nullptr, mutex_, {
    if (name == nullptr) {
      name = GetDefaultPool(ctx.device_type);
    }
    ptr.reset(MemoryPool::Create(name));
    ptr->Init(ctx, allow_missing);
  });
  return ptr;
}

}  // namespace memory_pool
}  // namespace mnm
