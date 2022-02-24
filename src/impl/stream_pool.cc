/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/stream_pool.cc
 * \brief RAF stream pool underlying implementation
 */
#include <mutex>
#include <string>
#include <unordered_map>
#include "raf/device_api.h"
#include "raf/registry.h"
#include "raf/stream_pool.h"

namespace raf {
namespace stream_pool {

using device_api::DeviceAPI;
using registry::PerDeviceStore;

int Tag::GetTagIndex_(const std::string& tag) {
  static std::mutex mutex;
  static std::unordered_map<std::string, int> map;
  std::lock_guard<std::mutex> lock(mutex);
  if (map.count(tag)) {
    return map[tag];
  }
  int value = map.size();
  map[tag] = value;
  return value;
}

class Stream::Impl {
 public:
  explicit Impl(const Device& dev) : device(dev), api(DeviceAPI::Get(dev.device_type())) {
    this->stream = api->CreateStream(dev);
  }

  ~Impl() {
    if (stream != nullptr && api != nullptr) {
      api->FreeStream(device, stream);
    }
  }

 public:
  Device device;
  std::shared_ptr<DeviceAPI> api;
  void* stream;
};

class StreamPool {
 public:
  explicit StreamPool(const Device& dev) : device(dev), api(DeviceAPI::Get(dev.device_type())) {
  }

  ~StreamPool() {
    for (auto& i : pool) {
      for (auto& j : i) {
        j = nullptr;
      }
    }
  }

  std::shared_ptr<Stream> GetStream(int tag_index, int index) {
    std::lock_guard<std::mutex> lock(mutex);
    if (tag_index >= static_cast<int>(pool.size())) {
      pool.resize(tag_index + 1);
    }
    if (index >= static_cast<int>(pool[tag_index].size())) {
      pool[tag_index].resize(index + 1);
    }
    if (pool[tag_index][index] == nullptr) {
      pool[tag_index][index] = std::make_shared<Stream>(new Stream::Impl(device));
    }
    return pool[tag_index][index];
  }

 public:
  static std::shared_ptr<StreamPool> Get(const Device& dev) {
    static PerDeviceStore<StreamPool, false>* per_device = new PerDeviceStore<StreamPool, false>();
    std::shared_ptr<StreamPool>& ret = per_device->Get(dev);
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(per_device->mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<StreamPool>(dev);
      }
    }
    return ret;
  }

 public:
  Device device;
  std::shared_ptr<DeviceAPI> api;
  std::vector<std::vector<std::shared_ptr<Stream>>> pool;
  std::mutex mutex;
};

Stream::Stream(Stream::Impl* impl) : impl(impl) {
}

Stream::~Stream() = default;

void* Stream::data() const {
  return impl ? impl->stream : nullptr;
}

void Stream::Wait() const {
  impl->api->WaitStream(data());
}

std::shared_ptr<Stream> Stream::Get(const Device& dev, int tag_index, int index) {
  return StreamPool::Get(dev)->GetStream(tag_index, index);
}

}  // namespace stream_pool
}  // namespace raf
