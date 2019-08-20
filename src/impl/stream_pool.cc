#include <mutex>
#include <string>
#include <unordered_map>

#include <mnm/device_api.h>
#include <mnm/registry.h>
#include <mnm/stream_pool.h>

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
  Impl(const Context& ctx) : ctx(ctx), api(DeviceAPI::Get(ctx.device_type)) {
    api->SetDevice(ctx.device_id);
    this->stream = api->CreateStream();
  }

  ~Impl() {
    if (stream != nullptr && api != nullptr) {
      api->SetDevice(ctx.device_id);
      api->FreeStream(stream);
    }
  }

 public:
  Context ctx;
  std::shared_ptr<DeviceAPI> api;
  void* stream;
};

class StreamPool {
 public:
  StreamPool(const Context& ctx) : ctx(ctx), api(DeviceAPI::Get(ctx.device_type)) {
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
    // static auto per_device = std::make_shared<PerContextStore<StreamPool, false>>();
    std::shared_ptr<StreamPool>& ret = per_device->Get(ctx);
    if (ret == nullptr) {
      auto lock = per_device->GrabLock();
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

std::shared_ptr<Stream> Stream::Get(const Context& ctx, int tag_index, int index) {
  return StreamPool::Get(ctx)->GetStream(tag_index, index);
}

}  // namespace stream_pool
}  // namespace mnm
