/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/event_pool.cc
 * \brief RAF event pool underlying implementation
 */
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include "raf/device_api.h"
#include "raf/registry.h"
#include "raf/event_pool.h"

namespace raf {
namespace event_pool {

using device_api::DeviceAPI;
using registry::PerDeviceStore;

class Event::Impl {
 public:
  explicit Impl(Device dev, uint32_t flags, void* event)
      : device_(std::move(dev)), flags_(flags), event_(event) {
  }

  ~Impl() {
    if (event_ != nullptr) {
      EventPool::Get(device_)->RecycleEvent(flags_, event_);
    }
  }

 public:
  Device device_;
  uint32_t flags_;
  void* event_;
};

Event::Event(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

void* Event::data() const {
  return impl_ ? impl_->event_ : nullptr;
}

EventPool::~EventPool() {
  for (auto& pr : freed_events_) {
    for (void* event : pr.second) {
      api_->FreeEvent(device_, event);
    }
  }
}

std::shared_ptr<Event> EventPool::GetEvent(uint32_t flags) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& pool = freed_events_[flags];
  void* event;
  if (!pool.empty()) {
    event = pool.back();
    pool.pop_back();
  } else {
    event = api_->CreateEvent(device_, flags);
  }
  return std::shared_ptr<Event>(new Event(std::make_unique<Event::Impl>(device_, flags, event)));
}

EventPool::EventPool(const Device& dev) : device_(dev), api_(DeviceAPI::Get(dev.device_type())) {
}

void EventPool::RecycleEvent(uint32_t flags, void* event) {
  std::lock_guard<std::mutex> lock(mutex_);
  freed_events_[flags].push_back(event);
}

std::shared_ptr<EventPool> EventPool::Get(const Device& dev) {
  static auto* per_device = new PerDeviceStore<EventPool, false>();
  std::shared_ptr<EventPool>& ret = per_device->Get(dev);
  if (ret == nullptr) {
    std::lock_guard<std::mutex> lock(per_device->mutex_);
    if (ret == nullptr) {
      ret = std::shared_ptr<EventPool>(new EventPool(dev));
    }
  }
  return ret;
}

Event::~Event() = default;

}  // namespace event_pool
}  // namespace raf
