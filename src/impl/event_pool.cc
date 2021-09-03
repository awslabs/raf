/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/event_pool.cc
 * \brief MNM event pool underlying implementation
 */
#include <mutex>
#include <string>
#include <unordered_map>
#include "mnm/device_api.h"
#include "mnm/registry.h"
#include "mnm/event_pool.h"

namespace mnm {
namespace event_pool {

using device_api::DeviceAPI;
using registry::PerDeviceStore;

class Event::Impl {
 public:
  explicit Impl(const Device& dev, uint32_t flags)
      : device(dev), api(DeviceAPI::Get(dev.device_type())) {
    this->event = api->CreateEvent(dev, flags);
  }

  ~Impl() {
    if (event != nullptr && api != nullptr) {
      api->FreeEvent(device, event);
    }
  }

 public:
  Device device;
  uint32_t flags;
  std::shared_ptr<DeviceAPI> api;
  void* event;
};

Event::Event(Event::Impl* impl) : impl(impl) {
}

Event::Event(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {
}

void* Event::data() const {
  return impl ? impl->event : nullptr;
}

class EventPool {
 public:
  explicit EventPool(const Device& dev) : device(dev), api(DeviceAPI::Get(dev.device_type())) {
  }

  ~EventPool() {
  }

  std::shared_ptr<Event> GetEvent(uint32_t flags) {
    std::lock_guard<std::mutex> lock(mutex);
    auto& pool = freed_events[flags];
    if (!pool.empty()) {
      auto ret = std::make_shared<Event>(std::move(pool.back()));
      pool.pop_back();
      return ret;
    } else {
      return std::make_shared<Event>(new Event::Impl(device, flags));
    }
  }

  void RecycleEvent(std::unique_ptr<Event::Impl> event) {
    std::lock_guard<std::mutex> lock(mutex);
    freed_events[event->flags].push_back(std::move(event));
  }

 public:
  static std::shared_ptr<EventPool> Get(const Device& dev) {
    static auto* per_device = new PerDeviceStore<EventPool, false>();
    std::shared_ptr<EventPool>& ret = per_device->Get(dev);
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(per_device->mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<EventPool>(dev);
      }
    }
    return ret;
  }

 public:
  Device device;
  std::shared_ptr<DeviceAPI> api;
  std::unordered_map<uint32_t, std::vector<std::unique_ptr<Event::Impl>>> freed_events;
  std::mutex mutex;
};

std::shared_ptr<Event> Event::Create(const Device& dev, uint32_t flags) {
  return EventPool::Get(dev)->GetEvent(flags);
}

Event::~Event() {
  if (impl != nullptr) {
    auto pool = EventPool::Get(impl->device);
    pool->RecycleEvent(std::move(impl));
  }
};

}  // namespace event_pool
}  // namespace mnm
