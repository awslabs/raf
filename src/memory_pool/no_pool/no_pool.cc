/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/memory_pool/no_pool/no_pool.cc
 * \brief No memory pool
 */
#include <atomic>
#include "raf/device.h"
#include "raf/device_api.h"
#include "raf/memory_pool.h"
#include "raf/registry.h"

namespace raf {
namespace memory_pool {
namespace no_pool {

using device_api::DeviceAPI;

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
  std::shared_ptr<DeviceAPI> api;
};

class NonOwnedAsyncMemory final : public Memory {
 public:
  explicit NonOwnedAsyncMemory(void* data, void* stream, const Device& dev,
                               std::shared_ptr<DeviceAPI> api) {
    this->data = data;
    this->stream = stream;
    this->device = dev;
    this->api = std::move(api);
  }

  ~NonOwnedAsyncMemory() {
    if (data != nullptr) {
      api->FreeMemoryAsync(data, stream);
    }
  }

 public:
  std::shared_ptr<DeviceAPI> api;
  void* stream;
};

class NoPool final : public MemoryPool {
 public:
  explicit NoPool(Device dev) {
    this->device = dev;
    this->api = DeviceAPI::Get(dev.device_type());

    if (dev.device_type() == DevType::kCUDA()) {
      this->api->SetDevice(dev.device_id());
    }
  }

  std::string GetName() {
    return "no_pool";
  }

  int64_t GetAllocBytes(int64_t nbytes) override {
    return nbytes;
  }

  std::shared_ptr<Memory> Alloc(int64_t nbytes, int64_t alignment) override {
    CHECK_GE(nbytes, 0);
    void* data = nullptr;
    if (nbytes > 0) {
      data = api->AllocMemory(nbytes, alignment);
    }
    return std::make_shared<NonOwnedMemory>(data, device, api);
  }

  std::shared_ptr<Memory> AllocAsync(int64_t nbytes, void* stream,
                                     int64_t alignment = kDefaultMemoryAlignment) override {
    CHECK_GE(nbytes, 0);
    void* data = nullptr;
    if (nbytes > 0) {
      data = api->AllocMemoryAsync(nbytes, stream, alignment);
    }
    return std::make_shared<NonOwnedAsyncMemory>(data, stream, device, api);
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
    auto ret = api->GetPoolSize();
    return {BytesToMegaBytes(ret.first), BytesToMegaBytes(ret.second)};
  }

 public:
  static void* make(const Device& dev) {
    return new NoPool(dev);
  }

  Device device;
  std::shared_ptr<DeviceAPI> api;
};

RAF_REGISTER_GLOBAL("raf.memory_pool._make.no_pool").set_body_typed([](const Device& dev) {
  return NoPool::make(dev);
});

}  // namespace no_pool
}  // namespace memory_pool
}  // namespace raf
