/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/memory_pool/page_unit_pool/page_unit_pool.cc
 * \brief A memory pool that use page as memory unit
 */
#include <atomic>
#include <tvm/relay/transform.h>
#include "raf/device_api.h"
#include "raf/memory_pool.h"
#include "raf/registry.h"

namespace raf {
namespace memory_pool {
namespace page_unit_pool {

using device_api::DeviceAPI;

/*!
 * \brief A wrapper which holds the a chunck of memory that owned by nobody.
 * The memory chunck hold by this object could be any sizes.
 *
 * The memory could locate on cpu, gpu or any other devices, determined by the device api.
 *
 * \sa NonOwnedMemory
 */
class NonOwnedMemory final : public Memory {
 public:
  explicit NonOwnedMemory(void* data, const Device& dev, std::shared_ptr<DeviceAPI> api) {
    this->data = data;
    this->device = dev;
    this->api = std::move(api);
  }

  ~NonOwnedMemory() {
    if (data != nullptr) {
      api->FreeMemory(data);
    }
  }

 public:
  /*! \brief The pointer to the DeviceAPI which determines the context of memory. */
  std::shared_ptr<DeviceAPI> api;
};

/*!
 * \brief A Memory Pool that organizes the Memory as multiple memory pages. The default size of
 * memory page is 4KB. A memory chunck hold by NonOwnedMemory is composed of one/multiple pages.
 *
 * In this pool, all memory chunck are divide into multiple groups by the number of memory pages.
 * When user request a chunck of memory with size N, the pool will first find whether there is
 * available memory chunck with the same size in this pool. If so, return this available chunck. If
 * not, allocate a new memory chunck with size N, and return it.
 *
 * As the pool hold a reference to each memory chunck allocate, thus the memory chunck won't be
 * freed once it is allocated, until user's application finishes or fails.
 *
 * \example Assume the Page Size is 4KB. When user requests a chunck of memory with size 2KB, the
 * user will actually get a memory chunck with size 4KB, wrapped in NonOwnedMemory.
 *
 * \sa PageUnitPool
 */
class PageUnitPool : public MemoryPool {
 public:
  explicit PageUnitPool(Device dev, int64_t pool_limit = 0) {
    this->device = dev;
    this->api = DeviceAPI::Get(dev.device_type());
    this->max_pool_size = pool_limit;

    if (dev.device_type() == DevType::kCUDA()) {
      this->api->SetDevice(dev.device_id());
    }
  }

  std::string GetName() {
    return "page_unit_pool";
  }

  int64_t GetAllocBytes(int64_t nbytes) override {
    // round the chunck size to mutlpile page size.
    return !!(nbytes & ((1 << page_size_exp) - 1)) + (nbytes >> page_size_exp) << page_size_exp;
  }

  virtual inline void* AllocDeviceMemory(int64_t nbytes, int64_t alignment) {
    try {
      return api->AllocMemory(nbytes, alignment);
    } catch (const dmlc::Error& e) {
      return nullptr;
    }
  }

  int64_t FreeUnusedChunks() {
    // Remove the memory from the pool and return the freed memory in bytes.
    // Since this is the last share_ptr, the removed memory will be deconstructed and freed.
    int64_t total_free = 0;
    curr_pool_size = 0;

    std::vector<int64_t> page_nbytes;
    for (auto kv : _pool) {
      page_nbytes.push_back(kv.first);
    }

    for (auto nbytes : page_nbytes) {
      auto nchunk = _pool[nbytes].size();
      _pool[nbytes].remove_if([](std::shared_ptr<Memory>& mem) { return mem.use_count() == 1; });
      total_free += nbytes * (nchunk - _pool[nbytes].size());
      curr_pool_size += nbytes * _pool[nbytes].size();
    }
    return total_free;
  }

  std::shared_ptr<Memory> Alloc(int64_t nbytes, int64_t alignment) override {
    nbytes = GetAllocBytes(nbytes);
    CHECK_GE(nbytes, 0);

    // Find whether there are available memory chuncks in the pool.
    // If so, return the available memory chunck.
    if (_pool.find(nbytes) == _pool.end()) {
      _pool.insert({nbytes, std::list<std::shared_ptr<Memory>>()});
    }
    for (const auto& it : _pool[nbytes]) {
      if (it.use_count() == 1) {
        int64_t address = (int64_t)it->data;
        if (address % alignment == 0) return it;
      }
    }

    // If not, allocate a new memory chunck from device.
    void* data = nullptr;
    if (nbytes > 0) {
      data = AllocDeviceMemory(nbytes, alignment);

      // Out of memory or exceed the user-specified limitation, free unused chunks on other pages.
      size_t free_nbytes = SIZE_MAX;
      if ((max_pool_size > 0 && curr_pool_size >= max_pool_size) ||
          (data == nullptr && free_nbytes > 0)) {
        free_nbytes = FreeUnusedChunks();
        DLOG(WARNING) << "Failed to allocate " << BytesToMegaBytes(nbytes)
                      << " MBs). Ran GC and got " << BytesToMegaBytes(free_nbytes) << " more MBs";
      }

      // Re-allocate the desired chunk if needed.
      if (data == nullptr && free_nbytes > 0) {
        data = AllocDeviceMemory(nbytes, alignment);
      }
      if (data == nullptr) {
        // If the freed memory is insufficient, then we can do nothing in memory pool.
        size_t used, allocated;
        std::tie(used, allocated) = GetPoolSize();
        LOG(FATAL) << "Out-Of-Memory. Tried to allocate " << BytesToMegaBytes(nbytes)
                   << " MBs; Already allocated " << allocated << " MBs and used " << used << " MBs";
        throw;
      }
      curr_pool_size += nbytes;
      std::shared_ptr<Memory> new_mem = std::make_shared<NonOwnedMemory>(data, device, api);
      _pool[nbytes].push_back(new_mem);
      return new_mem;
    } else {
      return std::make_shared<NonOwnedMemory>(data, device, api);
    }
  }

  std::shared_ptr<Memory> AllocAsync(int64_t nbytes, void* stream,
                                     int64_t alignment = kDefaultMemoryAlignment) override {
    LOG(FATAL) << "Please use NoPool to use AllocAsync.";
    throw;
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

  std::pair<float, float> GetPoolSize() override {
    // First query the device API and use its numbers if available.
    auto ret = api->GetPoolSize();
    float used_total = BytesToMegaBytes(ret.first);
    float pool_total = BytesToMegaBytes(ret.second);

    if (used_total == 0 && pool_total == 0) {
      // First get all chunk sizes. Note that we do not count the number of used chunks
      // with use_count > 1 here becuase the kv itself also holds a share_ptr.
      std::vector<int64_t> page_nbytes;
      for (auto kv : _pool) {
        page_nbytes.push_back(kv.first);
      }

      // Then directly access each page to get the precise use_count.
      for (auto nbytes : page_nbytes) {
        size_t used_chunks = 0;
        for (const auto& it : _pool[nbytes]) {
          used_chunks += (it.use_count() > 1) ? 1 : 0;
        }
        used_total += BytesToMegaBytes(nbytes * used_chunks);
        pool_total += BytesToMegaBytes(nbytes * _pool[nbytes].size());
      }
    }
    return std::make_pair(used_total, pool_total);
  }

 public:
  static void* make(const Device& dev) {
    int64_t max_pool_limit = 0;
    if (const char* val = getenv("RAF_MEMORY_POOL_SIZE_LIMIT")) {
      max_pool_limit = atol(val);
    }
    return new PageUnitPool(dev, max_pool_limit);
  }

 protected:
  Device device;
  /*! \brief The size of each memory page (exponent). */
  static const int64_t page_size_exp = 12;
  /*! \brief The current pool size in bytes. */
  int64_t curr_pool_size = 0;
  /*! \brief The maximum allowed size (bytes) in the pool. 0 means no limit. */
  int64_t max_pool_size = 0;
  /*! \brief The pointer to the DeviceAPI which determines the context of memory. */
  std::shared_ptr<DeviceAPI> api;
  /*! \brief The pool that hold the references to NonOwnedMemory. */
  std::unordered_map<int64_t, std::list<std::shared_ptr<Memory>>> _pool;
};

RAF_REGISTER_GLOBAL("raf.memory_pool._make.page_unit_pool").set_body_typed([](const Device& dev) {
  return PageUnitPool::make(dev);
});

}  // namespace page_unit_pool
}  // namespace memory_pool
}  // namespace raf
