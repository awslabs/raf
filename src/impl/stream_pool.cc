/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/tensor.cc
 * \brief MNM stream pool underlying implementation
 */
#include <mutex>
#include <string>
#include <unordered_map>
#include "mnm/device_api.h"
#include "mnm/registry.h"
#include "mnm/stream_pool.h"

namespace mnm {
namespace stream_pool {

using device_api::DeviceAPI;
using registry::PerContextStore;

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
  explicit Impl(const Context& ctx) : ctx(ctx), api(DeviceAPI::Get(ctx.device_type)) {
    this->stream = api->CreateStream(ctx);
  }

  ~Impl() {
    if (stream != nullptr && api != nullptr) {
      api->FreeStream(ctx, stream);
    }
  }

 public:
  Context ctx;
  std::shared_ptr<DeviceAPI> api;
  void* stream;
};

class StreamPool {
 public:
  explicit StreamPool(const Context& ctx) : ctx(ctx), api(DeviceAPI::Get(ctx.device_type)) {
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
      pool[tag_index][index] = std::make_shared<Stream>(new Stream::Impl(ctx));
    }
    return pool[tag_index][index];
  }

 public:
  static std::shared_ptr<StreamPool> Get(const Context& ctx) {
    static PerContextStore<StreamPool, false>* per_device =
        new PerContextStore<StreamPool, false>();
    std::shared_ptr<StreamPool>& ret = per_device->Get(ctx);
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(per_device->mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<StreamPool>(ctx);
      }
    }
    return ret;
  }

 public:
  Context ctx;
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
  impl->api->WaitStream(impl->ctx, data());
}

std::shared_ptr<Stream> Stream::Get(const Context& ctx, int tag_index, int index) {
  return StreamPool::Get(ctx)->GetStream(tag_index, index);
}

}  // namespace stream_pool
}  // namespace mnm
