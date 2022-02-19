/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/memory_pool/no_pool/no_pool.cc
 * \brief No memory pool
 */
#include <atomic>
#include "mnm/device.h"
#include "mnm/device_api.h"
#include "mnm/memory_pool.h"
#include "mnm/registry.h"

namespace mnm {
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

MNM_REGISTER_GLOBAL("mnm.memory_pool._make.no_pool").set_body_typed([](const Device& dev) {
  return NoPool::make(dev);
});

}  // namespace no_pool
}  // namespace memory_pool
}  // namespace mnm
